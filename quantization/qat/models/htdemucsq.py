# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Simon Rouard.
"""
This code contains the spectrogram and Hybrid version of Demucs.
"""
import math
import numpy as np
import typing as tp
import torch
from torch import nn
from torch.nn import functional as F
from fractions import Fraction
from einops import rearrange
import random
from demucs.states import capture_init
from demucs.spec import spectro, ispectro
from quantization.qat.models.demucsq import DConv, LayerScale, rescale_module
from quantization.qat.models.hdemucsq import pad1d, ScaledEmbedding, HEncLayer, MultiWrap, HDecLayer
from quantization.qat.qat_layers import Add, Mul, Const
from quantization.qat.qat_utils import quantize_modules, replace_encoderq, replace_decoderq
from process import preprocess, postprocess

def create_sin_embedding(length: int, dim: int, shift: int = 0, device="cpu", max_period=10000):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)
    )
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1:: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe[None, :].to(device)


def create_sin_embedding_cape(
    length: int,
    dim: int,
    batch_size: int,
    mean_normalize: bool,
    augment: bool,  # True during training
    max_global_shift: float = 0.0,  # delta max
    max_local_shift: float = 0.0,  # epsilon max
    max_scale: float = 1.0,
    device: str = "cpu",
    max_period: float = 10000.0,
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = 1.0 * torch.arange(length).view(-1, 1, 1)  # (length, 1, 1)
    pos = pos.repeat(1, batch_size, 1)  # (length, batch_size, 1)
    if mean_normalize:
        pos -= torch.nanmean(pos, dim=0, keepdim=True)

    if augment:
        delta = np.random.uniform(
            -max_global_shift, +max_global_shift, size=[1, batch_size, 1]
        )
        delta_local = np.random.uniform(
            -max_local_shift, +max_local_shift, size=[length, batch_size, 1]
        )
        log_lambdas = np.random.uniform(
            -np.log(max_scale), +np.log(max_scale), size=[1, batch_size, 1]
        )
        pos = (pos + delta + delta_local) * np.exp(log_lambdas)

    pos = pos.to(device)

    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    ).float()


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.const = Const()

    def forward(self, x):
        """
        x: (B, T, C)
        if num_groups=1: Normalisation on all T and C together for each B
        """
        x = x.transpose(1, 2)
        return self.const(super().forward(x).transpose(1, 2))


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        group_norm=0,
        norm_first=False,
        norm_out=False,
        layer_norm_eps=1e-5,
        layer_scale=False,
        init_values=1e-4,
        device=None,
        dtype=None,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        auto_sparsity=False,
        sparsity=0.95,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.activation = nn.Identity()
        if self.activation_relu_or_gelu==1:
            self.activation = nn.ReLU()
        elif self.activation_relu_or_gelu==2:
            self.activation = nn.GELU()
        self.add_norm1 = Add()
        self.add_norm2 = Add()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        if batch_first = False, src shape is (T, B, C)
        the case where batch_first=True is not covered
        """
        x = src
        if self.norm_first:
            x = self.add_norm1(x, self.gamma_1(self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)))
            x = self.add_norm2(x, self.gamma_2(self._ff_block(self.norm2(x))))

            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(self.add_norm1(x,self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask))))
            x = self.norm2(self.add_norm2(x, self.gamma_2(self._ff_block(x))))

        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: bool = False,
        norm_out: bool = False,
        sparse=False,
        mask_type="diag",
        mask_random_seed=42,
        sparse_attn_window=500,
        global_window=50,
        sparsity=0.95,
        auto_sparsity=None,
        device=None,
        dtype=None,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.sparse = sparse
        self.auto_sparsity = auto_sparsity
        if sparse:
            if not auto_sparsity:
                self.mask_type = mask_type
                self.sparse_attn_window = sparse_attn_window
                self.global_window = global_window
            self.sparsity = sparsity

        self.cross_attn: nn.Module
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1: nn.Module
        self.norm2: nn.Module
        self.norm3: nn.Module
        if group_norm:
            self.norm1 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = MyGroupNorm(int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if activation == F.relu:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.add_norm1 = Add()
        self.add_norm2 = Add()

    def forward(self, q, k, mask=None):
        """
        Args:
            q: tensor of shape (T, B, C)
            k: tensor of shape (S, B, C)
            mask: tensor of shape (T, S)

        """
        if self.norm_first:
            x = self.add_norm1(q, self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask)))
            x = self.add_norm2(x, self.gamma_2(self._ff_block(self.norm3(x))))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(self.add_norm1(q, self.gamma_1(self._ca_block(q, k, mask))))
            x = self.norm2(self.add_norm2(x, self.gamma_2(self._ff_block(x))))

        return x

    # self-attention block
    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: bool = False,
        group_norm: int = False,
        norm_first: bool = False,
        norm_out: bool = False,
        max_period: float = 10000.0,
        weight_decay: float = 0.0,
        lr: tp.Optional[float] = None,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        weight_pos_embed: float = 1.0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: list = [5000.0, 1.0, 1.4],
        sparse_self_attn: bool = False,
        sparse_cross_attn: bool = False,
        mask_type: str = "diag",
        mask_random_seed: int = 42,
        sparse_attn_window: int = 500,
        global_window: int = 50,
        auto_sparsity: bool = False,
        sparsity: float = 0.95,
    ):
        super().__init__()
        """
        """
        assert dim % num_heads == 0

        hidden_dim = int(dim * hidden_scale)

        self.num_layers = num_layers
        # classic parity = 1 means that if idx%2 == 1 there is a
        # classical encoder else there is a cross encoder
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_decay = weight_decay
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)

        self.lr = lr

        activation: tp.Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        self.norm_in_t: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        # spectrogram layers
        self.layers = nn.ModuleList()
        # temporal layers
        self.layers_t = nn.ModuleList()

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "mask_type": mask_type,
            "mask_random_seed": mask_random_seed,
            "sparse_attn_window": sparse_attn_window,
            "global_window": global_window,
            "sparsity": sparsity,
            "auto_sparsity": auto_sparsity,
            "batch_first": True,
        }

        kwargs_classic_encoder = dict(kwargs_common)
        kwargs_classic_encoder.update({
            "sparse": sparse_self_attn,
        })
        kwargs_cross_encoder = dict(kwargs_common)
        kwargs_cross_encoder.update({
            "sparse": sparse_cross_attn,
        })

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:

                self.layers.append(MyTransformerEncoderLayer(**kwargs_classic_encoder))
                self.layers_t.append(
                    MyTransformerEncoderLayer(**kwargs_classic_encoder)
                )

            else:
                self.layers.append(CrossTransformerEncoderLayer(**kwargs_cross_encoder))

                self.layers_t.append(
                    CrossTransformerEncoderLayer(**kwargs_cross_encoder)
                )

        self.add_x = Add()
        self.add_xt = Add()
        self.const_pos_emb_2d = Const()
        self.const_pos_emb = Const()

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape
        pos_emb_2d = self.const_pos_emb_2d(create_2d_sin_embedding(
            C, Fr, T1, x.device, self.max_period
        ))  # (1, C, Fr, T1)
        pos_emb_2d = rearrange(pos_emb_2d, "b c fr t1 -> b (t1 fr) c")
        x = rearrange(x, "b c fr t1 -> b (t1 fr) c")
        x = self.norm_in(x)
        x = self.add_x(x, self.weight_pos_embed * pos_emb_2d)

        B, C, T2 = xt.shape
        xt = rearrange(xt, "b c t2 -> b t2 c")  # now T2, B, C
        pos_emb = self.const_pos_emb(self._get_pos_embedding(T2, B, C, x.device))
        pos_emb = rearrange(pos_emb, "t2 b c -> b t2 c")
        xt = self.norm_in_t(xt)
        xt = self.add_xt(xt, self.weight_pos_embed * pos_emb)

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = rearrange(x, "b (t1 fr) c -> b c fr t1", t1=T1)
        xt = rearrange(xt, "b t2 c -> b c t2")
        return x, xt

    def _get_pos_embedding(self, T, B, C, device):
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            pos_emb = create_sin_embedding(
                T, C, shift=shift, device=device, max_period=self.max_period
            )
        elif self.emb == "cape":
            if self.training:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=self.cape_augment,
                    max_global_shift=self.cape_glob_loc_scale[0],
                    max_local_shift=self.cape_glob_loc_scale[1],
                    max_scale=self.cape_glob_loc_scale[2],
                )
            else:
                pos_emb = create_sin_embedding_cape(
                    T,
                    C,
                    B,
                    device=device,
                    max_period=self.max_period,
                    mean_normalize=self.cape_mean_normalize,
                    augment=False,
                )

        elif self.emb == "scaled":
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embeddings(pos)[:, None]

        return pos_emb

    def make_optim_group(self):
        group = {"params": list(self.parameters()), "weight_decay": self.weight_decay}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


class HTDemucsQ(nn.Module):
    """
    Spectrogram and hybrid Demucs model.
    The spectrogram model has the same structure as Demucs, except the first few layers are over the
    frequency axis, until there is only 1 frequency, and then it moves to time convolutions.
    Frequency layers can still access information across time steps thanks to the DConv residual.

    Hybrid model have a parallel time branch. At some layer, the time branch has the same stride
    as the frequency branch and then the two are combined. The opposite happens in the decoder.

    Models can either use naive iSTFT from masking, Wiener filtering ([Ulhih et al. 2017]),
    or complex as channels (CaC) [Choi et al. 2020]. Wiener filtering is based on
    Open Unmix implementation [Stoter et al. 2019].

    The loss is always on the temporal domain, by backpropagating through the above
    output methods and iSTFT. This allows to define hybrid models nicely. However, this breaks
    a bit Wiener filtering, as doing more iteration at test time will change the spectrogram
    contribution, without changing the one from the waveform, which will lead to worse performance.
    I tried using the residual option in OpenUnmix Wiener implementation, but it didn't improve.
    CaC on the other hand provides similar performance for hybrid, and works naturally with
    hybrid models.

    This model also uses frequency embeddings are used to improve efficiency on convolutions
    over the freq. axis, following [Isik et al. 2020] (https://arxiv.org/pdf/2008.04470.pdf).

    Unlike classic Demucs, there is no resampling here, and normalization is always applied.
    """

    @capture_init
    def __init__(
        self,
        sources,
        # Channels
        audio_channels=2,
        channels=48,
        channels_time=None,
        growth=2,
        # STFT
        nfft=4096,
        wiener_iters=0,
        end_iters=0,
        wiener_residual=False,
        cac=True,
        # Main structure
        depth=4,
        rewrite=True,
        # Frequency branch
        multi_freqs=None,
        multi_freqs_depth=3,
        freq_emb=0.2,
        emb_scale=10,
        emb_smooth=True,
        # Convolutions
        kernel_size=8,
        time_stride=2,
        stride=4,
        context=1,
        context_enc=0,
        # Normalization
        norm_starts=4,
        norm_groups=4,
        # DConv residual branch
        dconv_mode=1,
        dconv_depth=2,
        dconv_comp=8,
        dconv_init=1e-3,
        # Before the Transformer
        bottom_channels=0,
        # Transformer
        t_layers=5,
        t_emb="sin",
        t_hidden_scale=4.0,
        t_heads=8,
        t_dropout=0.0,
        t_max_positions=10000,
        t_norm_in=True,
        t_norm_in_group=False,
        t_group_norm=False,
        t_norm_first=True,
        t_norm_out=True,
        t_max_period=10000.0,
        t_weight_decay=0.0,
        t_lr=None,
        t_layer_scale=True,
        t_gelu=True,
        t_weight_pos_embed=1.0,
        t_sin_random_shift=0,
        t_cape_mean_normalize=True,
        t_cape_augment=True,
        t_cape_glob_loc_scale=[5000.0, 1.0, 1.4],
        t_sparse_self_attn=False,
        t_sparse_cross_attn=False,
        t_mask_type="diag",
        t_mask_random_seed=42,
        t_sparse_attn_window=500,
        t_global_window=100,
        t_sparsity=0.95,
        t_auto_sparsity=False,
        # ------ Particuliar parameters
        t_cross_first=False,
        # Weight init
        rescale=0.1,
        # Metadata
        samplerate=44100,
        segment=10,
        use_train_segment=True,
    ):
        """
        Args:
            sources (list[str]): list of source names.
            n_splitter: number of splitter channels
            n_combiner: number of combiner channels
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            channels_time: if not None, use a different `channels` value for the time branch.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            wiener_iters: when using Wiener filtering, number of iterations at test time.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            wiener_residual: add residual source before wiener filtering.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            multi_freqs: list of frequency ratios for splitting frequency bands with `MultiWrap`.
            multi_freqs_depth: how many layers to wrap with `MultiWrap`. Only the outermost
                layers will be wrapped.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            time_stride: stride for the final time layer, after the merge.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            bottom_channels: if >0 it adds a linear layer (1x1 Conv) before and after the
                transformer in order to change the number of channels
            t_layers: number of layers in each branch (waveform and spec) of the transformer
            t_emb: "sin", "cape" or "scaled"
            t_hidden_scale: the hidden scale of the Feedforward parts of the transformer
                for instance if C = 384 (the number of channels in the transformer) and
                t_hidden_scale = 4.0 then the intermediate layer of the FFN has dimension
                384 * 4 = 1536
            t_heads: number of heads for the transformer
            t_dropout: dropout in the transformer
            t_max_positions: max_positions for the "scaled" positional embedding, only
                useful if t_emb="scaled"
            t_norm_in: (bool) norm before addinf positional embedding and getting into the
                transformer layers
            t_norm_in_group: (bool) if True while t_norm_in=True, the norm is on all the
                timesteps (GroupNorm with group=1)
            t_group_norm: (bool) if True, the norms of the Encoder Layers are on all the
                timesteps (GroupNorm with group=1)
            t_norm_first: (bool) if True the norm is before the attention and before the FFN
            t_norm_out: (bool) if True, there is a GroupNorm (group=1) at the end of each layer
            t_max_period: (float) denominator in the sinusoidal embedding expression
            t_weight_decay: (float) weight decay for the transformer
            t_lr: (float) specific learning rate for the transformer
            t_layer_scale: (bool) Layer Scale for the transformer
            t_gelu: (bool) activations of the transformer are GeLU if True, ReLU else
            t_weight_pos_embed: (float) weighting of the positional embedding
            t_cape_mean_normalize: (bool) if t_emb="cape", normalisation of positional embeddings
                see: https://arxiv.org/abs/2106.03143
            t_cape_augment: (bool) if t_emb="cape", must be True during training and False
                during the inference, see: https://arxiv.org/abs/2106.03143
            t_cape_glob_loc_scale: (list of 3 floats) if t_emb="cape", CAPE parameters
                see: https://arxiv.org/abs/2106.03143
            t_sparse_self_attn: (bool) if True, the self attentions are sparse
            t_sparse_cross_attn: (bool) if True, the cross-attentions are sparse (don't use it
                unless you designed really specific masks)
            t_mask_type: (str) can be "diag", "jmask", "random", "global" or any combination
                with '_' between: i.e. "diag_jmask_random" (note that this is permutation
                invariant i.e. "diag_jmask_random" is equivalent to "jmask_random_diag")
            t_mask_random_seed: (int) if "random" is in t_mask_type, controls the seed
                that generated the random part of the mask
            t_sparse_attn_window: (int) if "diag" is in t_mask_type, for a query (i), and
                a key (j), the mask is True id |i-j|<=t_sparse_attn_window
            t_global_window: (int) if "global" is in t_mask_type, mask[:t_global_window, :]
                and mask[:, :t_global_window] will be True
            t_sparsity: (float) if "random" is in t_mask_type, t_sparsity is the sparsity
                level of the random part of the mask.
            t_cross_first: (bool) if True cross attention is the first layer of the
                transformer (False seems to be better)
            rescale: weight rescaling trick
            use_train_segment: (bool) if True, the actual size that is used during the
                training is used during inference.
        """
        super().__init__()
        self.cac = cac
        self.set_splitter_combiner(1,1)
        self.wiener_residual = wiener_residual
        self.audio_channels = audio_channels
        self.sources = sources
        self.n_srcs = len(sources)
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.channels = channels
        self.samplerate = samplerate
        self.segment = segment
        self.use_train_segment = use_train_segment
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.wiener_iters = wiener_iters
        self.end_iters = end_iters
        self.freq_emb = None
        assert wiener_iters == end_iters

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.tencoder = nn.ModuleList()
        self.tdecoder = nn.ModuleList()

        chin = audio_channels
        chin_z = chin  # number of channels for the freq branch
        if self.cac:
            chin_z *= 2
        chout = channels_time or channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                assert freqs == 1
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            kw = {
                "kernel_size": ker,
                "stride": stri,
                "freq": freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }
            kwt = dict(kw)
            kwt["freq"] = 0
            kwt["kernel_size"] = kernel_size
            kwt["stride"] = stride
            kwt["pad"] = True
            kw_dec = dict(kw)
            multi = False
            if multi_freqs and index < multi_freqs_depth:
                multi = True
                kw_dec["context_freq"] = False

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z, chout_z, dconv=dconv_mode & 1, context=context_enc, **kw
            )
            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=dconv_mode & 1,
                    context=context_enc,
                    empty=last_freq,
                    **kwt
                )
                self.tencoder.append(tenc)

            if multi:
                enc = MultiWrap(enc, multi_freqs)
            self.encoder.append(enc)
            if index == 0:
                chin = self.audio_channels * len(self.sources)
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(
                chout_z,
                chin_z,
                dconv=dconv_mode & 2,
                last=index == 0,
                context=context,
                **kw_dec
            )
            if multi:
                dec = MultiWrap(dec, multi_freqs)
            if freq:
                tdec = HDecLayer(
                    chout,
                    chin,
                    dconv=dconv_mode & 2,
                    empty=last_freq,
                    last=index == 0,
                    context=context,
                    **kwt
                )
                self.tdecoder.insert(0, tdec)
            self.decoder.insert(0, dec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride
            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale
                )
                self.freq_emb_scale = freq_emb
                self.add_freq = Add()
                self.mul_freq = Mul()

        if rescale:
            rescale_module(self, reference=rescale)

        transformer_channels = channels * growth ** (depth - 1)
        if bottom_channels:
            self.channel_upsampler = nn.Conv1d(transformer_channels, bottom_channels, 1)
            self.channel_downsampler = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )
            self.channel_upsampler_t = nn.Conv1d(
                transformer_channels, bottom_channels, 1
            )
            self.channel_downsampler_t = nn.Conv1d(
                bottom_channels, transformer_channels, 1
            )

            transformer_channels = bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                emb=t_emb,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                cross_first=t_cross_first,
                dropout=t_dropout,
                max_positions=t_max_positions,
                norm_in=t_norm_in,
                norm_in_group=t_norm_in_group,
                group_norm=t_group_norm,
                norm_first=t_norm_first,
                norm_out=t_norm_out,
                max_period=t_max_period,
                weight_decay=t_weight_decay,
                lr=t_lr,
                layer_scale=t_layer_scale,
                gelu=t_gelu,
                sin_random_shift=t_sin_random_shift,
                weight_pos_embed=t_weight_pos_embed,
                cape_mean_normalize=t_cape_mean_normalize,
                cape_augment=t_cape_augment,
                cape_glob_loc_scale=t_cape_glob_loc_scale,
                sparse_self_attn=t_sparse_self_attn,
                sparse_cross_attn=t_sparse_cross_attn,
                mask_type=t_mask_type,
                mask_random_seed=t_mask_random_seed,
                sparse_attn_window=t_sparse_attn_window,
                global_window=t_global_window,
                sparsity=t_sparsity,
                auto_sparsity=t_auto_sparsity,
            )
        else:
            self.crosstransformer = None

    def _spec(self, x):
        hl = self.hop_length
        nfft = self.nfft
        x0 = x  # noqa

        # We re-pad the signal in order to keep the property
        # that the size of the output is exactly the size of the input
        # divided by the stride (here hop_length), when divisible.
        # This is achieved by padding by 1/4th of the kernel size (here nfft).
        # which is not supported by torch.stft.
        # Having all convolution operations follow this convention allow to easily
        # align the time and frequency branches later on.
        assert hl == nfft // 4
        le = int(math.ceil(x.shape[-1] / hl))
        pad = hl // 2 * 3
        x = pad1d(x, (pad, pad + le * hl - x.shape[-1]), mode="reflect")

        z = spectro(x, nfft, hl)[..., :-1, :]
        assert z.shape[-1] == le + 4, (z.shape, x.shape, le)
        z = z[..., 2: 2 + le]
        return z

    def _ispec(self, z, length=None, scale=0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad
        x = ispectro(z, hl, length=le)
        x = x[..., pad: pad + length]
        return x

    def _magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        if self.cac:
            B, C, Fr, T = z.shape
            m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
            m = m.reshape(B, C * 2, Fr, T)
        else:
            m = z.abs()
        return m

    def _mask(self, m):
        B, S, C, Fr, T = m.shape
        out = m.view(B, S, -1, 2, Fr, T).permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out

    def valid_length(self, length: int):
        """
        Return a length that is appropriate for evaluation.
        In our case, always return the training length, unless
        it is smaller than the given length, in which case this
        raises an error.
        """
        if not self.use_train_segment:
            return length
        training_length = int(self.segment * self.samplerate)
        if training_length < length:
            raise ValueError(
                    f"Given length {length} is longer than "
                    f"training length {training_length}")
        return training_length


    def pre_process(self, mix):
        self.length = mix.shape[-1]
        self.length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                self.training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < self.training_length:
                    self.length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, self.training_length - self.length_pre_pad))
        z = self._spec(mix)
        mag = self._magnitude(z)
        x = mag

        # unlike previous Demucs, we always normalize because it is easier.
        # x will be the freq. branch input.
        self.mean = x.mean(dim=(1, 2, 3), keepdim=True)
        self.std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - self.mean) / (1e-5 + self.std)

        # Prepare the time branch input.
        xt = mix
        self.meant = xt.mean(dim=(1, 2), keepdim=True)
        self.stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - self.meant) / (1e-5 + self.stdt)

        # --------------
        # Pre-process
        # --------------
        x = preprocess(x, n_splitter=self.n_splitter)
        xt = preprocess(xt, n_splitter=self.n_splitter, normalize=False)

        return x, xt


    def post_process(self, x, xt):

        # --------------
        # Post-process
        # --------------
        x = postprocess(x, n_combiner=self.n_combiner)
        xt = postprocess(xt, n_combiner=self.n_combiner)

        x = x * self.std[:, None] + self.mean[:, None]
        xt = xt * self.stdt[:, None] + self.meant[:, None]

        zout = self._mask(x)
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, self.length)
            else:
                x = self._ispec(zout, self.training_length)
        else:
            x = self._ispec(zout, self.length)

        if self.use_train_segment:
            if self.training:
                xt = xt.view(self.B, self.n_srcs, -1, self.length)
            else:
                xt = xt.view(self.B, self.n_srcs, -1, self.training_length)
        else:
            xt = xt.view(self.B, self.n_srcs, -1, self.length)

        x = xt + x
        if self.length_pre_pad:
            x = x[..., :self.length_pre_pad]
        return x

    def forward(self, mix):
        # --------------------------
        # Pre-Process
        # --------------------------
        x, xt = self.pre_process(mix)
        self.B, C, Fq, T = x.shape

        # --------------------------
        # Model
        # --------------------------
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = self.add_freq(x, self.mul_freq(emb, self.freq_emb_scale))

            saved.append(x)
        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        x = x.reshape(self.n_combiner, self.B, self.n_srcs, -1, Fq, T)
        xt = xt.reshape(self.n_combiner, self.B, self.n_srcs, -1, xt.shape[-1])

        # --------------------------
        # Post-Process
        # --------------------------
        y = self.post_process(x, xt)

        return y

    def set_splitter_combiner(self, n_splitter, n_combiner):
        self.n_splitter = n_splitter
        self.n_combiner = n_combiner

    def quantize_model(self, gradient_based=True,
                       weight_quant=True, weight_n_bits=8,
                       act_quant=True, act_n_bits=8, inout_nl_quant=False,
                       in_quant=False, in_act_n_bits=8,
                       out_quant=True, out_act_n_bits=8):

        params_dict = {'gradient_based': gradient_based, 'act_quant': act_quant, 'weight_quant': weight_quant,
                       'weight_n_bits': weight_n_bits, 'act_n_bits': act_n_bits}

        for n, m in self.named_modules():
            if type(m) == HEncLayer:
                if n in ['encoder.0','tencoder.0']:
                    replace_encoderq(m, ['conv','gelu'], {'n_splitter':self.n_splitter,
                                                           'gradient_based': gradient_based,
                                                           'act_quant': act_quant,
                                                           'inout_nl_quant': inout_nl_quant,
                                                           'weight_quant': weight_quant,
                                                           'in_quant': in_quant,
                                                           'weight_n_bits': weight_n_bits,
                                                           'act_n_bits': act_n_bits,
                                                           'in_act_n_bits': in_act_n_bits})
                else:
                    quantize_modules(m, ['conv','gelu'], params_dict)
                quantize_modules(m, ['rewrite','glu'], params_dict)
            elif type(m) == HDecLayer:
                quantize_modules(m, ['rewrite','glu'], params_dict)
                quantize_modules(m, ['add'], params_dict)
                if m.last:
                    replace_decoderq(m, ['conv_tr'], {'n_combiner': self.n_combiner,
                                                     'gradient_based': gradient_based,
                                                     'act_quant': act_quant,
                                                     'inout_nl_quant': inout_nl_quant,
                                                     'act_n_bits': out_act_n_bits,
                                                     'out_quant': out_quant,
                                                     'out_act_n_bits': out_act_n_bits,
                                                     'weight_quant': weight_quant,
                                                     'weight_n_bits': weight_n_bits,
                                                     'train_res_dec': n in ['decoder.3']})
                else:
                    quantize_modules(m, ['conv_tr','gelu'], params_dict)
            elif type(m) == HTDemucsQ:
                quantize_modules(m, ['channel_downsampler'], params_dict)
                quantize_modules(m, ['channel_downsampler_t'], params_dict)
                quantize_modules(m, ['channel_upsampler'], params_dict)
                quantize_modules(m, ['channel_upsampler_t'], params_dict)
                quantize_modules(m, ['add_freq'], params_dict)
                quantize_modules(m, ['mul_freq'], params_dict)
                quantize_modules(m.freq_emb, ['embedding'], params_dict)
                quantize_modules(m.freq_emb, ['mul'], params_dict)
            elif type(m) == DConv:
                for layer in m.layers:
                    quantize_modules(layer, ['0','1','2'], params_dict)
                    quantize_modules(layer, ['3','4','5'], params_dict)
                    quantize_modules(layer[6], ['mul'], params_dict)
                for i in range(len(m.adds)):
                    quantize_modules(m.adds, [str(i)], params_dict)
            elif type(m) == CrossTransformerEncoder:
                quantize_modules(m, ['norm_in'], params_dict)
                quantize_modules(m, ['norm_in_t'], params_dict)
                quantize_modules(m, ['add_x'], params_dict)
                quantize_modules(m, ['add_xt'], params_dict)
                quantize_modules(m, ['const_pos_emb'], params_dict)
                quantize_modules(m, ['const_pos_emb_2d'], params_dict)
            elif type(m) == CrossTransformerEncoderLayer:
                quantize_modules(m, ['add_norm1'], params_dict)
                quantize_modules(m, ['add_norm2'], params_dict)
                quantize_modules(m, ['norm1'], params_dict)
                quantize_modules(m, ['norm2'], params_dict)
                quantize_modules(m, ['norm3'], params_dict)
                quantize_modules(m.norm_out, ['const'], params_dict)
                quantize_modules(m.gamma_1, ['mul'], params_dict)
                quantize_modules(m.gamma_2, ['mul'], params_dict)
                quantize_modules(m, ['linear1', 'activation'], params_dict)
                quantize_modules(m, ['linear2'], params_dict)
                quantize_modules(m, ['cross_attn'], params_dict)
            elif type(m) == MyTransformerEncoderLayer:
                quantize_modules(m, ['add_norm1'], params_dict)
                quantize_modules(m, ['add_norm2'], params_dict)
                quantize_modules(m, ['norm1'], params_dict)
                quantize_modules(m, ['norm2'], params_dict)
                quantize_modules(m.norm_out, ['const'], params_dict)
                quantize_modules(m.gamma_1, ['mul'], params_dict)
                quantize_modules(m.gamma_2, ['mul'], params_dict)
                quantize_modules(m, ['linear1', 'activation'], params_dict)
                quantize_modules(m, ['linear2'], params_dict)
                quantize_modules(m, ['self_attn'], params_dict)



