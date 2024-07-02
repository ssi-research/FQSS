import os
import torch
import torch.nn as nn
import math
from quantization.qat.qat_layers import Add, Mul, Const
from quantization.qat.qat_utils import quantize_modules, replace_encoderq, replace_decoderq
from process import preprocess, postprocess

EPS_T = 1e-6
EPS = 1e-8


class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).
    """

    def __init__(self, input_size, max_len=2500, device='cpu'):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False, device=device)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )
        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.const = Const()

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.const(self.pe[:, : x.size(1)].clone().detach())


class TransformerLayer(nn.Module):
    def __init__(
            self,
            n_filters,
            n_ffn,
            n_heads,
            dropout=0.0,
    ):
        super(TransformerLayer, self).__init__()
        self.mha = nn.MultiheadAttention(n_filters, n_heads, dropout=dropout, batch_first=False)
        self.ffn = nn.Sequential(nn.Linear(n_filters, n_ffn),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(n_ffn, n_filters))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(n_filters, eps=EPS_T)
        self.norm2 = nn.LayerNorm(n_filters, eps=EPS_T)

    def forward(self, x):
        # -----------------------
        # Norm
        # -----------------------
        x_norm1 = self.norm1(x)
        # -----------------------
        # Multi-head attention
        # -----------------------
        query = x_norm1.permute(1, 0, 2)
        x_mha = self.mha(query, query, query)[0]
        x_mha = x_mha.permute(1, 0, 2)
        # -----------------------
        # Dropout + Norm
        # -----------------------
        x = x + self.dropout1(x_mha)
        x_norm2 = self.norm2(x)
        # -----------------------
        # POS + FFN
        # -----------------------
        x_norm2 = x_norm2.permute(1, 0, 2)
        x_ffn = self.ffn(x_norm2)
        x_ffn = x_ffn.permute(1, 0, 2)
        # -----------------------
        # Dropout
        # -----------------------
        out = x + self.dropout2(x_ffn)
        return out


class TransformerBlock(nn.Module):
    def __init__(
            self,
            n_filters,
            n_heads,
            n_ffn,
            num_layers=8,
            dropout=0.0,
            device='cpu',
    ):
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(TransformerLayer(n_filters, n_heads=n_heads, n_ffn=n_ffn, dropout=dropout))
        self.norm = nn.LayerNorm(n_filters, eps=EPS_T)
        self.pos = PositionalEncoding(n_filters, device=device)
        self.pos_add = Add()


    def forward(self, x):
        x_pos = self.pos(x)
        x_trans = self.pos_add(x, x_pos)
        for layer in self.layers:
            x_trans = layer(x_trans)
        out = self.norm(x_trans)
        return out


class DualPathBlock(nn.Module):
    def __init__(
        self,
        n_filters,
        n_heads,
        n_ffn,
        dropout=0.0,
        device='cpu',
    ):
        super(DualPathBlock, self).__init__()
        self.intra_transformer_block = TransformerBlock(n_filters=n_filters, n_heads=n_heads, n_ffn=n_ffn, dropout=dropout, device=device)
        self.inter_transformer_block = TransformerBlock(n_filters=n_filters, n_heads=n_heads, n_ffn=n_ffn, dropout=dropout, device=device)
        self.intra_norm = nn.GroupNorm(num_groups=1, num_channels=n_filters, eps=EPS)
        self.inter_norm = nn.GroupNorm(num_groups=1, num_channels=n_filters, eps=EPS)
        self.intra_add = Add()
        self.inter_add = Add()

    def forward(self, x):
        B, F, K, S = x.shape
        # --------------------
        # intra RNN
        # --------------------
        # [BS, K, F]
        intra = x.permute(0, 3, 2, 1).contiguous().reshape(B * S, K, F)
        # [BS, K, F]
        intra = self.intra_transformer_block(intra)
        # [B, S, K, F]
        intra = intra.reshape(B, S, K, F)
        # [B, F, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        intra = self.intra_norm(intra)
        intra = self.intra_add(intra,x)

        # --------------------
        # inter RNN
        # --------------------
        # [B, K, S, F]
        inter = intra.permute(0, 2, 3, 1).contiguous()
        # [BK, S, F]
        inter = inter.reshape(B * K, S, F)
        # [BK, S, F]
        inter = self.inter_transformer_block(inter)
        # [B, K, S, F]
        inter = inter.reshape(B, K, S, F)
        # [B, F, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        inter = self.inter_norm(inter)
        # [B, F, K, S]
        out = self.inter_add(inter,intra)
        return out


class MaskGenerator(nn.Module):
    def __init__(
        self,
        n_srcs: int,
        n_filters: int,
        n_repeats: int = 2,
        n_heads: int = 8,
        chunk_size: int = 250,
        n_ffn: int = 1024,
        dropout: float = 0.0,
        device: str = 'cpu',
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.chunk_size = chunk_size

        self.norm = nn.GroupNorm(num_groups=1, num_channels=n_filters, eps=EPS)
        self.conv1d = nn.Conv1d(n_filters, n_filters, 1, bias=False)

        # DP Blocks
        self.layers = nn.ModuleList([])
        for x in range(n_repeats):
            self.layers.append(DualPathBlock(n_filters,
                                             n_heads=n_heads,
                                             n_ffn=n_ffn,
                                             dropout=dropout,
                                             device=device))

        # Gating and masking in 2D
        self.conv2d = nn.Conv2d(n_filters, n_srcs * n_filters, kernel_size=1, bias=True)
        self.end_conv = nn.Sequential(nn.Conv1d(n_filters, n_filters, 1, bias=False), nn.ReLU())
        self.prelu = nn.PReLU()
        self.net_out = nn.Sequential(nn.Conv1d(n_filters, n_filters, 1, bias=True), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(n_filters, n_filters, 1, bias=True), nn.Sigmoid())
        self.mul = Mul()

    def padding(self, x, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        x : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = x.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            #pad = torch.Tensor(torch.zeros(B, N, gap)).type(x.type())
            pad = torch.zeros(B, N, gap, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=2)

        #_pad = torch.Tensor(torch.zeros(B, N, P)).type(x.type())
        _pad = torch.zeros(B, N, P, device=x.device, dtype=x.dtype)
        x = torch.cat([_pad, x, _pad], dim=2)

        return x, gap

    def segmentation(self, x, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        x : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, F, M = x.shape
        P = K // 2
        x, gap = self.padding(x, K)
        # [B, N, K, S]
        input1 = x[:, :, :-P].contiguous().view(B, F, -1, K)
        input2 = x[:, :, P:].contiguous().view(B, F, -1, K)
        x = (torch.cat([input1, input2], dim=3).view(B, F, -1, K).transpose(2, 3))
        return x.contiguous(), gap

    def over_add(self, x, gap):
        """Merge the sequence with the overlap-and-add method.
        Arguments
        ---------
        x : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = x.shape
        P = K // 2
        # [B, N, S, K]
        x = x.transpose(2, 3).contiguous().view(B, N, -1, K * 2)
        input1 = x[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = x[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        x = input1 + input2
        # [B, N, L]
        if gap > 0:
            x = x[:, :, :-gap]

        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate separation mask.
        Args:
            x (torch.Tensor): 3D Tensor with shape [batch, features, frames]

        Returns:
            Tensor: shape [batch, num_sources, features, frames]
        """
        B, F, _ = x.shape

        # Norm + Conv1D
        x_norm = self.norm(x)  # [B, F, M]
        x_conv1d = self.conv1d(x_norm)

        # Segmentation
        x_segment, gap = self.segmentation(x_conv1d, self.chunk_size)

        # ----------------------------------------------
        # Main inter and intra transformers blocks
        # ----------------------------------------------
        for layer in self.layers:
            x_segment = layer(x_segment)

        # Conv2D + PRelu
        x_conv2d = self.conv2d(self.prelu(x_segment))
        B, _, _, L = x_conv2d.shape
        x_conv2d = x_conv2d.reshape(B * self.n_srcs, -1, self.chunk_size, L)

        # Over and Add
        x_over_add = self.over_add(x_conv2d, gap)
        out = self.end_conv(self.mul(self.net_out(x_over_add), self.net_gate(x_over_add))) # [B*spks, F, L]

        # Reshape + Transpose
        _, _, L = out.shape
        out = out.reshape(B,self.n_srcs,F,L) # [B, n_srcs, F, L]
        return out


class SepformerQ(nn.Module):
    """Sepformer: separation model
    Paper: Attention is All You Need in Speech Separation (8 Mar 2021)
    https://arxiv.org/abs/2010.13154
    """

    def __init__(
        self,
        n_spks: int = 1,
        # encoder/decoder parameters
        kernel_size: int = 16,
        stride: int = 8,
        n_filters: int = 256,
        n_repeats: int = 2,
        n_heads: int = 8,
        chunk_size: int = 250,
        device: str = 'cpu',
    ):
        super().__init__()

        self.n_srcs = n_spks
        self.enc_num_feats = n_filters
        self.set_splitter_combiner(1,1)
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=1,
                                               out_channels=n_filters,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=0,
                                               bias=False), nn.ReLU())

        self.masker = MaskGenerator(
            n_spks,
            n_filters,
            n_repeats=n_repeats,
            n_heads=n_heads,
            chunk_size=chunk_size,
            device=device,
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=n_filters,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )

        self.mul = Mul()

    def pre_process(self, x):
        return preprocess(x, n_splitter=self.n_splitter)

    def post_process(self, x):
        return postprocess(x, n_combiner=self.n_combiner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform source separation
        """
        # B: batch size
        # L: input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources
        # E : enc_num_ch
        # D : dec_num_ch

        # ----------
        # Pre-process
        # ----------
        x = self.pre_process(x)

        batch_size = x.shape[0]

        # ----------
        # Encoder
        # ----------
        feats = self.encoder(x)  # [B, E, L] -> [B, F, M]

        # ----------
        # Mask
        # ----------
        masked = self.mul(self.masker(feats), feats.unsqueeze(1))  # [B, S, F, M]
        masked_reshaped = torch.reshape(masked, (batch_size * self.n_srcs, self.enc_num_feats, -1))  # [B*S, F, M]

        # ----------
        # Decoder
        # ----------
        out_decoder = self.decoder(masked_reshaped)  # [B*S, D, L]
        out = out_decoder.reshape((self.n_combiner, batch_size, self.n_srcs, 1, -1)) # [D, B, S, 1, L]

        # ----------
        # Post-process
        # ----------
        out = self.post_process(out)

        return out

    def load_pretrain(self, weights_path):
        model_state_dict = self.state_dict()
        if os.path.isfile(weights_path):
            model_state_dict_weights = torch.load(weights_path)
            model_state_dict_weights = model_state_dict_weights.get('state_dict', model_state_dict_weights)
            # Remove fmodel params from state_dict
            keys = list(model_state_dict_weights.keys())
            for key in keys:
                if key.startswith("fmodel."):
                    model_state_dict_weights.pop(key)
            assert len(model_state_dict.keys()) == len(model_state_dict_weights.keys()), "Error: mismatch models weights. Please check if the model configurations match to model weights!"
            for new_key, key in zip(model_state_dict.keys(), model_state_dict_weights.keys()):
                model_state_dict[new_key] = model_state_dict_weights.get(key)
        else:
            # Load encoder
            encoder_loaded = torch.load(weights_path + "/encoder.ckpt")
            model_state_dict['encoder.0.weight'] = encoder_loaded['conv1d.weight']
            # Load masknet
            masknet_loaded = torch.load(weights_path + "/masknet.ckpt")
            for new_key, key in zip(self.masker.state_dict().keys(), masknet_loaded):
                model_state_dict["masker." + new_key] = masknet_loaded.get(key)
            # Load decoder
            decoder_loaded = torch.load(weights_path + "/decoder.ckpt")
            model_state_dict['decoder.weight'] = decoder_loaded['weight']
        # Loading
        self.load_state_dict(model_state_dict, strict=True)

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
            if type(m) == SepformerQ:
                replace_encoderq(m.encoder, ['0','1'], {'n_splitter':self.n_splitter,
                                                       'gradient_based': gradient_based,
                                                       'act_quant':act_quant,
                                                       'inout_nl_quant': inout_nl_quant,
                                                       'weight_quant':weight_quant,
                                                       'in_quant':in_quant,
                                                       'weight_n_bits': weight_n_bits,
                                                       'act_n_bits': act_n_bits,
                                                       'in_act_n_bits': in_act_n_bits})
                replace_decoderq(m, ['decoder'], {'n_combiner':self.n_combiner,
                                                 'gradient_based': gradient_based,
                                                 'act_quant':act_quant,
                                                 'inout_nl_quant': inout_nl_quant,
                                                 'act_n_bits': act_n_bits,
                                                 'out_quant': out_quant,
                                                 'out_act_n_bits': out_act_n_bits,
                                                 'weight_quant':weight_quant,
                                                 'weight_n_bits': weight_n_bits,
                                                 'train_res_dec': True})
                quantize_modules(m, ['mul'], params_dict)
            elif type(m) == TransformerBlock:
                quantize_modules(m, ['norm'], params_dict)
                quantize_modules(m, ['pos_add'], params_dict)
                quantize_modules(m.pos, ['const'], params_dict)
            elif type(m) == TransformerLayer:
                quantize_modules(m, ['norm1'], params_dict)
                quantize_modules(m, ['norm2'], params_dict)
                quantize_modules(m, ['mha'], params_dict)
                quantize_modules(m.ffn, ['0'], params_dict)
                quantize_modules(m.ffn, ['1'], params_dict)
                quantize_modules(m.ffn, ['3'], params_dict)
            elif type(m) == DualPathBlock:
                quantize_modules(m, ['inter_norm'], params_dict)
                quantize_modules(m, ['intra_norm'], params_dict)
                quantize_modules(m, ['inter_add'], params_dict)
                quantize_modules(m, ['intra_add'], params_dict)
            elif type(m) == MaskGenerator:
                quantize_modules(m.net_out, ['0','1'], params_dict)
                quantize_modules(m.net_gate, ['0','1'], params_dict)
                quantize_modules(m, ['norm'], params_dict)
                quantize_modules(m, ['conv1d'], params_dict)
                quantize_modules(m, ['conv2d'], params_dict)
                quantize_modules(m.end_conv, ['0','1'], params_dict)
                quantize_modules(m, ['prelu'], params_dict)
                quantize_modules(m, ['mul'], params_dict)