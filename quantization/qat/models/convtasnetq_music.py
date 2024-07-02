import math
import torch
import torch.nn as nn
from quantization.qat.qat_layers import Add, Mul
from quantization.qat.qat_utils import quantize_modules, replace_encoderq, replace_decoderq
from process import preprocess, postprocess

EPS = 1e-8

def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.reshape(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes, device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().reshape(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.reshape(*outer_dimensions, -1)
    return result


class ChannelWiseLayerNorm(nn.Module):
    """
    Channel wise layer normalization
    """
    def __init__(self, N, eps=EPS):
        super().__init__()
        self.norm = nn.LayerNorm(N, eps=eps)

    def forward(self, x):
        """
        x: N x C x T
        """
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = self.norm(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class MaskGenerator(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, mask_act='relu'):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            mask_act: use which non-linear function to generate mask
        """
        super(MaskGenerator, self).__init__()
        # Hyper-parameter
        self.C = C
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelWiseLayerNorm(N, eps=EPS)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation // 2
                blocks += [ConvBlock(B, H, P, stride=1, padding=padding, dilation=dilation)]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)

        if mask_act == "sigmoid":
            mask_activate_layer = nn.Sigmoid()
        elif mask_act == "relu":
            mask_activate_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {mask_act}")

        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net, mask_conv1x1, mask_activate_layer)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        est_mask = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        est_mask = est_mask.reshape(M, self.C, N, K)  # [M, C*N, K] -> [M, C, N, K]
        return est_mask


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation):
        super(ConvBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=EPS)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)
        self.add = Add()

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        return self.add(out,residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation):
        super(DepthwiseSeparableConv, self).__init__()

        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)

        prelu = nn.PReLU()
        norm = nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=EPS)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class ConvTasNetMusicQ(nn.Module):
    def __init__(self,
                 sources=['drums', 'bass', 'other', 'vocals'],
                 audio_channels=2,
                 n_filters=256,
                 kernel=20,
                 stride=10,
                 bn_chan=256,
                 hid_chan=512,
                 conv_kernel=3,
                 n_blocks=10,
                 n_repeats=4,
                 mask_act='relu'):

        super(ConvTasNetMusicQ, self).__init__()
        self.sources = sources
        self.n_srcs = len(sources)
        self.set_splitter_combiner(1,1)
        self.stride = stride
        self.audio_channels = audio_channels
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=audio_channels,
                                               out_channels=n_filters,
                                               kernel_size=kernel,
                                               stride=stride,
                                               padding=0,
                                               bias=False), nn.ReLU())

        self.separator = MaskGenerator(n_filters,
                                         bn_chan,
                                         hid_chan,
                                         conv_kernel,
                                         n_blocks,
                                         n_repeats,
                                         self.n_srcs,
                                         mask_act)

        self.decoder = nn.Linear(n_filters,
                                 audio_channels * kernel,
                                 bias=False)

        self.mul = Mul()

    def pre_process(self, x):
        return preprocess(x, n_splitter=self.n_splitter, normalize=False)

    def post_process(self, x):
        return postprocess(x, n_combiner=self.n_combiner)

    def forward(self, x):

        # ----------
        # Pre-process
        # ----------
        x = self.pre_process(x)

        batch_size = x.shape[0]

        # ----------
        # Encoder
        # ----------
        feats = self.encoder(x)

        # ----------
        # Mask
        # ----------
        masked = self.mul(self.separator(feats), feats.unsqueeze(1))  # [B, n_srcs, n_filters, K]
        masked_transposed = torch.transpose(masked, 2, 3)  # [B, n_srcs, K, n_filters]

        # ----------
        # Decoder
        # ----------
        out_decoder = self.decoder(masked_transposed)  # [n_combiner, batch, n_srcs, K, audio_channels * kernel_size]
        K = out_decoder.shape[-2]


        out_decoder = out_decoder.reshape((self.n_combiner,
                                           batch_size,
                                           self.n_srcs,
                                           K,
                                           self.audio_channels,
                                           -1)).transpose(3, 4) # [n_combiner, batch, n_srcs, audio_channels, K, -1]

        out = overlap_and_add(out_decoder, self.stride) # [n_combiner, batch, n_srcs, audio_channels, -1]

        # ----------
        # Post-process
        # ----------
        out = self.post_process(out)

        return out

    def load_pretrain(self, weights_path):
        model_state_dict = self.state_dict()
        model_state_dict_weights = torch.load(weights_path)
        model_state_dict_weights = model_state_dict_weights.get('state_dict', model_state_dict_weights)
        # Remove fmodel params from state_dict
        keys = list(model_state_dict_weights.keys())
        for key in keys:
            if key.startswith("fmodel."):
                model_state_dict_weights.pop(key)
        assert len(model_state_dict.keys()) == len(model_state_dict_weights.keys()), "Error: mismatch models weights. Please check if the model configurations match to model weights!"
        for new_key, key in zip(model_state_dict.keys(), model_state_dict_weights.keys()):
            if 'beta' in key or 'gamma' in key:
                model_state_dict[new_key] = model_state_dict_weights.get(key).reshape(-1)
            else:
                model_state_dict[new_key] = model_state_dict_weights.get(key)
        self.load_state_dict(model_state_dict, strict=True)

    def set_splitter_combiner(self, n_splitter, n_combiner):
        self.n_splitter = n_splitter
        self.n_combiner = n_combiner

    def quantize_model(self,
                       gradient_based=True,
                       weight_quant=True, weight_n_bits=8,
                       act_quant=True, act_n_bits=8, inout_nl_quant=False,
                       in_quant=False, in_act_n_bits=8,
                       out_quant=True, out_act_n_bits=8):

        params_dict = {'gradient_based': gradient_based, 'act_quant': act_quant, 'weight_quant': weight_quant,
                        'weight_n_bits': weight_n_bits, 'act_n_bits': act_n_bits}

        for n, m in self.named_modules():
            if type(m) == ConvTasNetMusicQ:
                replace_encoderq(m.encoder, ['0','1'], {'n_splitter':self.n_splitter,
                                                     'gradient_based': gradient_based,
                                                     'act_quant':act_quant,
                                                     'act_n_bits': act_n_bits,
                                                     'inout_nl_quant': inout_nl_quant,
                                                     'weight_quant':weight_quant,
                                                     'weight_n_bits': weight_n_bits,
                                                     'in_quant':in_quant,
                                                     'in_act_n_bits': in_act_n_bits})
                replace_decoderq(m, ['decoder'], {'n_combiner':self.n_combiner,
                                                 'gradient_based': gradient_based,
                                                 'act_quant':act_quant,
                                                 'inout_nl_quant': inout_nl_quant,
                                                 'act_n_bits': out_act_n_bits,
                                                 'out_quant': out_quant,
                                                 'out_act_n_bits': out_act_n_bits,
                                                 'weight_quant':weight_quant,
                                                 'weight_n_bits': weight_n_bits,
                                                 'train_res_dec': False})
                quantize_modules(m, ['mul'], params_dict)
            elif type(m) == ConvBlock:
                quantize_modules(m.net, ['0', '1'], params_dict)
                quantize_modules(m.net, ['2'], params_dict)
                quantize_modules(m, ['add'], params_dict)
            elif type(m) == DepthwiseSeparableConv:
                quantize_modules(m.net, ['0', '1'], params_dict)
                quantize_modules(m.net, ['2'], params_dict)
                quantize_modules(m.net, ['3'], params_dict)
            elif type(m) == MaskGenerator:
                quantize_modules(m.network[0], ['norm'], params_dict)
                quantize_modules(m.network, ['1'], params_dict)
                quantize_modules(m.network, ['3','4'], params_dict)