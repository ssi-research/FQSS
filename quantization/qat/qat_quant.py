import numpy as np
import torch
import torch.nn as nn
from utils import get_device
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization import default_per_channel_weight_fake_quant, FakeQuantize
import math
DEVICE = get_device()


def MSE(x, y, weights):
    wl2 = torch.square(x - y)*weights/torch.sum(weights)
    return torch.sum(wl2)

class TorchWeightFakeQuantize(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        min_range = quantizer.min_range
        max_range = quantizer.max_range
        max_abs_range = torch.maximum(torch.abs(min_range), torch.abs(max_range))
        scales = max_abs_range / (2 ** (quantizer.n_bits - int(quantizer.sign)))
        zero_points = torch.zeros_like(scales)
        self.scales = scales.flatten()
        self.zero_points = zero_points.flatten()
        self.axis = quantizer.axis
        self.sign = quantizer.sign
        self.n_bits = quantizer.n_bits
    def forward(self, x):
        y = torch.fake_quantize_per_channel_affine(x,
                                                  scale=self.scales,
                                                  zero_point=self.zero_points,
                                                  axis=self.axis,
                                                  quant_min=-2**(self.n_bits-1) if self.sign else 0,
                                                  quant_max=2**(self.n_bits-1)-1 if self.sign else 2**self.n_bits-1)
        return y


class TorchActivationFakeQuantize(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        min_range = quantizer.min_range
        max_range = quantizer.max_range
        self.scale = float((max_range - min_range) / (2 ** quantizer.n_bits - 1))
        self.zero_point = int(torch.round(min_range / self.scale))
        self.zero_point = -self.zero_point if min_range < 0 else self.zero_point  # zp has to be positive, and a <=0, so we multiply by -1
        self.n_bits = quantizer.n_bits
    def forward(self, x):
        y = torch.fake_quantize_per_tensor_affine(x,
                                                 scale=self.scale,
                                                 zero_point=self.zero_point,
                                                 quant_min=0,
                                                 quant_max=2**self.n_bits-1)
        return y


class TorchDymActivationFakeQuantize(nn.Module):
    def __init__(self, quantizer):
        super().__init__()
        self.n_bits = quantizer.n_bits
        self.factor = quantizer.factor
    def forward(self, x):
        min_range = self.factor*x.min()
        max_range = self.factor*x.max()
        scale = float((max_range - min_range) / (2 ** self.n_bits - 1))
        zero_point = int(torch.round(min_range / scale))
        zero_point = -zero_point if min_range < 0 else zero_point  # zp has to be positive, and a <=0, so we multiply by -1
        y = torch.fake_quantize_per_tensor_affine(x,
                                                 scale=scale,
                                                 zero_point=zero_point,
                                                 quant_min=0,
                                                 quant_max=2**self.n_bits-1)
        return y


def activation_fake_quantize():
    return FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                  quant_min=0,
                                  quant_max=255,
                                  dtype=torch.quint8,
                                  qscheme=torch.per_tensor_affine,
                                  reduce_range=False)()


def weight_fake_quantize():
    return default_per_channel_weight_fake_quant()


def round_ste(x):
    return (torch.round(x) - x).detach() + x


def floor_ste(x):
    return (torch.floor(x) - x).detach() + x


def grad_sign(x, scale=1.0):
    x_scaled = x * scale
    return (torch.sign(x) - x_scaled).detach() + x_scaled


def grad_scale(x, scale):
    x_scaled = x * scale
    return (x - x_scaled).detach() + x_scaled


def clip_ste(x: torch.Tensor, min_val=-1.0, max_val=1.0):
    return (torch.clip(x, min=min_val, max=max_val) - x).detach() + x


def fix_range_to_include_zero(range_min: torch.Tensor, range_max: torch.Tensor, n_bits):
    min_positive = range_min > 0
    max_negative = range_max < 0
    mid_range = torch.logical_and(torch.logical_not(min_positive), torch.logical_not(max_negative))
    min_positive = min_positive.float()
    max_negative = max_negative.float()
    mid_range = mid_range.float()
    scale = (range_max - range_min) / (2 ** n_bits - 1)
    min_range_adj = scale * torch.round(range_min / scale)
    max_range_adj = range_max - range_min + min_range_adj
    min_range_adj = min_range_adj * mid_range + max_negative * range_min
    max_range_adj = max_range_adj * mid_range + min_positive * range_max
    return min_range_adj, max_range_adj


def linear_quantize(x, min_range, max_range, n_bits, sign=True, sym=False, scale_grad=False):
    if sym:
        # Symmetric quantizer
        Qmin = -2 ** (n_bits - 1) if sign else 0
        Qmax = 2 ** (n_bits - 1) - 1 if sign else 2**n_bits - 1
        max_abs_range = torch.maximum(torch.abs(min_range), torch.abs(max_range))
        delta = 2*max_abs_range / (2**n_bits - 1)
        scale_factor = 1.0 / math.sqrt((Qmax * max_abs_range.numel())) if scale_grad else 1.0
        delta = grad_scale(delta, scale_factor)
        X = round_ste(x / delta)
        y = delta * torch.clip(X, Qmin, Qmax)
    else:
        # Uniform quantizer
        Qmin = 0
        Qmax = 2**n_bits - 1
        delta = (max_range - min_range) / (2 ** n_bits - 1)
        n_channels = int(x.shape[-1])
        scale_factor = 1.0 / math.sqrt(Qmax * n_channels) if scale_grad else 1.0
        delta = grad_scale(delta, scale_factor)
        zp = min_range
        X = round_ste((x - zp)/ delta)
        y = delta * torch.clip(X, Qmin, Qmax) + zp
    return y


def mulaw_quantize(x, min_range, max_range, mu, n_bits, scale_grad):
    # Normalize to [-1,1]
    max_abs_range = torch.maximum(torch.abs(min_range), torch.abs(max_range))
    x_norm = x / max_abs_range

    # mu-law compression
    x_mu = grad_sign(x_norm)*torch.log1p(mu*torch.abs(x_norm))/torch.log1p(mu)
    # linear quantization
    x_mu_q = linear_quantize(x_mu, torch.Tensor([-1]).to(x.device),torch.Tensor([1]).to(x.device), n_bits, scale_grad=scale_grad)
    # mu-law decompression
    y_norm = grad_sign(x_mu_q)*(torch.pow(1+mu, torch.abs(x_mu_q))-1)/mu

    # Denormalize
    y = y_norm * max_abs_range
    return y


class GradientNlActivationFakeQuantize(nn.Module):
    """
    Per tensor non linear quanntization
    """
    def __init__(self, gradient_based, n_bits=8, scale_grad=False):
        super().__init__()
        self.n_bits = n_bits
        self.min_range = nn.Parameter(torch.Tensor([-0.5]), requires_grad=gradient_based)
        self.max_range = nn.Parameter(torch.Tensor([0.5]), requires_grad=gradient_based)
        self.mu = nn.Parameter(torch.Tensor([1.0]), requires_grad=gradient_based)
        self.max_observations = 50
        self.observer_mode = True
        self.alpha = 0.9
        self.n_iter = 0
        self.sign = True
        self.scale_grad = scale_grad

    def enable_observer(self, observer_mode):
        self.observer_mode = observer_mode

    def forward(self, x):
        if self.observer_mode and self.n_iter < self.max_observations:
            self.n_iter += 1
            tilde_range_max, tilde_range_min = x.max(), x.min()
            self.min_range.data = self.alpha*self.min_range+(1-self.alpha)*tilde_range_min
            self.max_range.data = self.alpha*self.max_range+(1-self.alpha)*tilde_range_max
            # if self.n_iter == self.max_observations:
            #     bias_correction = 1 - (self.alpha ** self.max_observations)
            #     self.min_range.data = self.min_range.data / bias_correction
            #     self.max_range.data = self.max_range.data / bias_correction
            return x

        assert self.max_range >= self.min_range, f"Min range is {self.min_range.item()} which is bigger than max range {self.max_range.item()}!"

        # Quantization
        y = mulaw_quantize(x, self.min_range, self.max_range, self.mu, self.n_bits, self.scale_grad)
        return y


class GradientActivationFakeQuantize(nn.Module):
    """
    Per tensor quanntization
    """
    def __init__(self, gradient_based, n_bits=8, sym=False, scale_grad=False):
        super().__init__()
        self.n_bits = n_bits
        self.sym = sym
        self.min_range = nn.Parameter(torch.Tensor([-0.5]), requires_grad=gradient_based)
        self.max_range = nn.Parameter(torch.Tensor([0.5]), requires_grad=gradient_based)
        self.max_observations = 50
        self.observer_mode = True
        self.alpha = 0.9
        self.n_iter = 0
        self.sign = True
        self.scale_grad = scale_grad

    def enable_observer(self, observer_mode):
        self.observer_mode = observer_mode
        self.sign = self.min_range.sign().item()<0

    def forward(self, x):
        if self.observer_mode and self.n_iter < self.max_observations:
            self.n_iter += 1
            tilde_range_max, tilde_range_min = x.max(), x.min()
            self.min_range.data = self.alpha*self.min_range+(1-self.alpha)*tilde_range_min
            self.max_range.data = self.alpha*self.max_range+(1-self.alpha)*tilde_range_max
            return x

        if self.training:
            self.sign = self.min_range.sign().item() < 0

        assert self.max_range >= self.min_range, f"Min range is {self.min_range.item()} which is bigger than max range {self.max_range.item()}!"

        # Quantization
        y = linear_quantize(x, self.min_range, self.max_range, self.n_bits, self.sign, self.sym, self.scale_grad)
        return y


class GradientActivationFakeQuantize_MSE(nn.Module):
    """
    Per tensor quanntization
    """
    def __init__(self, gradient_based, n_bits=8, sym=False, scale_grad=False):
        super().__init__()
        self.n_bits = n_bits
        self.sym = sym
        self.min_range = nn.Parameter(torch.Tensor([-0.5]), requires_grad=gradient_based)
        self.max_range = nn.Parameter(torch.Tensor([0.5]), requires_grad=gradient_based)
        self.max_observations = 50
        self.observer_mode = True
        self.alpha = 0.9
        self.n_iter = 0
        self.sign = True
        self.scale_grad = scale_grad
        self.hists = []
        self.hist_n_bins = 512

    def enable_observer(self, observer_mode):
        self.observer_mode = observer_mode
        self.sign = self.min_range.sign().item()<0

    def merge_hist(self):
        assert len(self.hists) > 0, "Error: missing histograms"
        merged_bins_min, merged_bins_max = np.inf, -np.inf
        merged_bin_width = np.inf
        for hist in self.hists:
            vals, bins = hist
            bin_min = np.min(bins)
            bin_max = np.max(bins)
            merged_bins_min = np.minimum(bin_min, merged_bins_min)
            merged_bins_max = np.maximum(bin_max, merged_bins_max)
            merged_bin_width = np.minimum(bins[1]-bins[0], merged_bin_width)

        merged_bins = np.arange(merged_bins_min, merged_bins_max + merged_bin_width, merged_bin_width)
        merged_vals = None
        for hist in self.hists:
            cumulative_hist = np.hstack([0, np.cumsum(hist[0])])
            cumulative_interpolated_hist = np.interp(merged_bins, hist[1], cumulative_hist)
            if merged_vals is None:
                merged_vals = np.diff(cumulative_interpolated_hist)
            else:
                merged_vals += np.diff(cumulative_interpolated_hist)
        return torch.from_numpy(merged_vals), torch.from_numpy(merged_bins[:-1])

    def mse_minmax_range(self, N=100):
        vals, bins = self.merge_hist()
        min_range, max_range = bins.min(), bins.max()
        delta = 0.5*(bins.max() - bins.min())/N
        best_min, best_max, best_error = min_range, max_range, torch.inf
        for i in range(0, N):
            min_value_i = min_range + delta*i
            for j in range(0, N):
                max_value_j = max_range - delta*j
                bins_quant = linear_quantize(bins, min_value_i, max_value_j, self.n_bits, self.sign, self.sym, self.scale_grad)
                error = MSE(bins, bins_quant, weights=vals)
                if error < best_error:
                    best_min, best_max, best_error = min_value_i, max_value_j, error
        return best_min, best_max

    def forward(self, x):
        if self.observer_mode:
            if self.n_iter < self.max_observations:
                self.n_iter += 1
                self.hists.append(np.histogram(x.detach().cpu().numpy(), bins=self.hist_n_bins))
                return x
            elif self.n_iter == self.max_observations:
                min_range, max_range = self.mse_minmax_range()
                self.min_range.data = min_range.reshape(1).float().to(DEVICE)
                self.max_range.data = max_range.reshape(1).float().to(DEVICE)
                self.observer_mode = False
                del self.hists

        if self.training:
            self.sign = self.min_range.sign().item() < 0

        assert self.max_range >= self.min_range, f"Min range is {self.min_range.item()} which is bigger than max range {self.max_range.item()}!"

        # Quantization
        y = linear_quantize(x, self.min_range, self.max_range, self.n_bits, self.sign, self.sym, self.scale_grad)
        return y


class DynamicActivationFakeQuantize(nn.Module):
    """
    Per tensor quanntization
    """
    def __init__(self, n_bits=8, sym=False, factor=0.99):
        super().__init__()
        self.n_bits = n_bits
        self.sym = sym
        self.factor = factor # mitigate outliers

    def forward(self, x):
        min_range = x.min()
        max_range = x.max()
        sign = min_range.sign().item() < 0
        if min_range == max_range:
            return x
        # Quantization
        y = linear_quantize(x, self.factor*min_range, self.factor*max_range, self.n_bits, sign, self.sym)
        return y


class GradientWeightFakeQuantize(nn.Module):
    """
    Per channel quanntization
    """
    def __init__(self, gradient_based, weight_shape, n_bits=8, sym=True, ch_out_idx=0, scale_grad=False):
        super().__init__()
        self.n_bits = n_bits
        self.sym = sym
        self.axis = ch_out_idx
        self.x_dims = list(range(len(weight_shape)))
        self.x_dims.remove(ch_out_idx)
        init_shape = [1]*len(weight_shape)
        init_shape[ch_out_idx] = weight_shape[ch_out_idx]
        self.min_range = nn.Parameter(-0.5*torch.ones(init_shape, device=DEVICE), requires_grad=gradient_based)
        self.max_range = nn.Parameter(0.5*torch.ones(init_shape, device=DEVICE), requires_grad=gradient_based)
        self.observer_mode = True
        self.sign = True
        self.scale_grad = scale_grad

    def enable_observer(self, observer_mode):
        self.observer_mode = observer_mode

    def forward(self, x):
        if self.observer_mode:
            self.max_range.data = torch.amax(x, dim=self.x_dims, keepdim=True)
            self.min_range.data = torch.amin(x, dim=self.x_dims, keepdim=True)
            self.observer_mode = False
            return x

        # Quantization
        y = linear_quantize(x, self.min_range, self.max_range, self.n_bits, self.sign, self.sym, self.scale_grad)
        return y


def get_activation_quantizer(gradient_based=True, nl=False, n_bits=8):
    if nl:
        return GradientNlActivationFakeQuantize(gradient_based, n_bits=n_bits)
    else:
        return GradientActivationFakeQuantize(gradient_based, n_bits=n_bits)


def get_weight_quantizer(gradient_based=True, weight_shape=(1,1,1), n_bits=8, ch_out_idx=0):
    return GradientWeightFakeQuantize(gradient_based, weight_shape, n_bits=n_bits, ch_out_idx=ch_out_idx)


def get_dym_activation_quantizer(n_bits=8, factor=0.99):
    return DynamicActivationFakeQuantize(n_bits=n_bits, factor=factor)
