import numpy as np
import torch
from torchmetrics import ScaleInvariantSignalNoiseRatio, SignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def quantize(x, threshold=1.0, n_bits=8, sign=True):
    delta = threshold / (2 ** (n_bits - int(sign)))
    min_val = -2 ** (n_bits - int(sign)) if sign else 0
    max_val = 2 ** (n_bits - int(sign)) - 1
    return torch.clip(torch.floor(x / delta), min_val, max_val) * delta

def preprocess(x, n_splitter=1, n_bits=8, sign=True, normalize=True):

    if len(x.shape) == 2: # 2D
        x = x.unsqueeze(1) # Output 3D: [batch, 1, samples]

    if n_splitter > 1:
        # Input > 2D: [batch, audio_channels, ...]
        if normalize:
            x, threshold = x / max(abs(x.min()), abs(x.max())), 1
        else:
            threshold = max(abs(x.min()), abs(x.max()))
        delta = threshold / (2 ** (n_bits - int(sign)))

        y = []
        for _ in range(n_splitter):
            x_quant = quantize(x, threshold=threshold, n_bits=n_bits, sign=sign)
            y.append(x_quant)
            # error=x-x_quant: The error is in range [0, delta]
            x = 2 * (x - x_quant) * threshold/delta - threshold  # make error in range [-threshold, threshold]
        return torch.cat(y, dim=1)  # Output : [batch, audio_channels*n_splitter, ...]

    return x

def postprocess(x, n_combiner=1, n_bits=8, sign=True):
    # Input shape: [n_combiner, batch, sources, audios_channels, ...]
    if n_combiner == 1:
        y = x.squeeze(0)
    else:
        delta = 1 / (2 ** (n_bits - int(sign)))
        y = x[0]
        for i in range(1,n_combiner):
            y += x[i] * (0.5 * delta) ** i

    if len(y.shape)<=4 and y.shape[-2] == 1: # audios_channels=1
        y = y.squeeze(-2)

    return y

def normalize_audio(waveform, dim=-1):
    return waveform / waveform.abs().max(dim=dim, keepdim=True)[0]

def max_clip(x, max_check, max_clip=0.9):
    x_max = torch.max(torch.abs(x))
    if x_max >= max_check:
        gain = max_clip/x_max
        x = x*gain
    return x

def calc_nsisdr(sig, ref, eps=1e-7):
    sig, ref = sig[0], ref[0]
    alpha = torch.dot(ref,sig)/torch.sum(ref**2)
    sisdr = (torch.sum(torch.square(alpha*ref)) + eps) / (torch.sum(torch.square(sig - alpha*ref)) + eps)
    return 10*np.log10(sisdr.item())

def calc_nsdr(ref, sig, eps=1e-7):
    """
    Compute the New SDR according to the MDX challenge definition: https://arxiv.org/pdf/2108.13559.pdf
    """
    sdr = (torch.sum(torch.square(ref)) + eps) / (torch.sum(torch.square(ref - sig)) + eps)
    return 10*np.log10(sdr.item())

def generate_2mix_snr(signal1, signal2, snr, clip=True):
    E1, E2 = torch.mean(signal1**2), torch.mean(signal2**2)
    if E1>0.0 and E2>0.0:
        current_snr = 10*np.log10(E1/E2)
        if current_snr < snr:
            gain2 = torch.sqrt((E1/E2)*(10**(-snr/10))) # decrease signal2
            signal2 = signal2*gain2
        else:
            gain1 = torch.sqrt((E2/E1)*(10**(snr/10))) # decrease signal1
            signal1 = signal1*gain1
    # Mixture
    mix = signal1 + signal2
    if clip:
        mix = max_clip(mix, max_check=0.9)
    return mix

def generate_3mix_snr(signal1, signal2, signal3, snr1_23, snr2_3):
    mix23 = generate_2mix_snr(signal2, signal3, snr2_3)
    mix = generate_2mix_snr(signal1, mix23, snr1_23)
    return mix

def generate_mix_noise(sig, noise, snr):
    Es, En = torch.mean(sig**2), torch.mean(noise**2)
    gain = torch.sqrt((Es/En)/(10**(snr/10))) if Es>0 else 1.0
    mix = sig + gain*noise
    mix = max_clip(mix, max_check=0.9)
    return mix

def swap_channel_order(sep_tensor, clean_tensor):
    n_src = clean_tensor.shape[0]
    if n_src == 1:
        return sep_tensor

    new_sep_tensor = sep_tensor.clone()
    for src in range(n_src):
        # The model output for specific src
        sep_ch = sep_tensor[src:src+1,:]
        # The order of the recovered clean signals is unknown and may not match to model output, so we match them by max SI-SNR
        max_sisnr, max_sisnr_idx = -torch.inf, 0
        for i in range(n_src):
            sisnr = ScaleInvariantSignalNoiseRatio()(sep_ch, clean_tensor[i])
            if sisnr > max_sisnr:
                max_sisnr = sisnr
                max_sisnr_idx = i
        # If swap occurs, signal is also swaped by signal sign, so we need to fix it
        new_sep_tensor[max_sisnr_idx,...] = sep_ch if src==max_sisnr_idx else -sep_ch
    return new_sep_tensor

def metric_evaluation(sep_waveform, clean_waveforms, sample_rate=16000):
    n_src = clean_waveforms.shape[0]
    sisnrs, sdrs, stois = np.zeros(n_src), np.zeros(n_src), np.zeros(n_src)

    for src in range(n_src):
        # The model output for specific src
        sep_waveform_ch = sep_waveform[src:src+1,:]

        # The order of the clean signals is unknown and may not match to model output, so we match them by max SI-SNR
        max_sisnr, max_sisnr_idx = -torch.inf, 0
        for i in range(n_src):
            sisnr = ScaleInvariantSignalNoiseRatio()(sep_waveform_ch, clean_waveforms[i])
            if sisnr > max_sisnr:
                max_sisnr = sisnr
                max_sisnr_idx = i
        clean_waveform_ch = clean_waveforms[max_sisnr_idx]

        # SI-SNR
        sisnr = max_sisnr
        # SDR
        sdr = SignalDistortionRatio()(sep_waveform_ch, clean_waveform_ch)
        # STOI
        stoi = ShortTimeObjectiveIntelligibility(fs=sample_rate)(sep_waveform_ch, clean_waveform_ch)
        # Store results
        sisnrs[src], sdrs[src], stois[src] = sisnr, sdr, stoi

    # Average by number of sources
    return np.mean(sisnrs), np.mean(sdrs), np.mean(stois)

def model_infer(model, mix, n_srcs=1, segment=None, overlap=0.25, device='cpu', target=None):

    if segment:
        channels, length = mix.shape
        num_srcs = model.n_srcs if hasattr(model,'n_srcs') else n_srcs
        out_shape = (num_srcs, channels, length) if channels>1 else (num_srcs, length)
        out = torch.zeros(*out_shape)
        sum_weight = torch.zeros(length)
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        weight = torch.cat([torch.arange(1, segment // 2 + 1), torch.arange(segment - segment // 2, 0, -1)])
        assert len(weight) == segment
        weight = (weight / weight.max())
        for offset in offsets:
            start = offset
            stop = min(start+segment, length)
            chunk = mix[...,start:stop]
            chunk_length = chunk.size(-1)
            chunk = F.pad(chunk, (0, segment - chunk_length))
            chunk_out = model_infer(model, chunk, device=device)
            chunk_out = chunk_out[...,:chunk_length]
            chunk_out = swap_channel_order(chunk_out, target[...,start:start+chunk_length]) if target is not None and num_srcs>1 else chunk_out
            # Overlap and Add (OLA)
            out[..., start:stop] += weight[:chunk_length] * chunk_out
            sum_weight[start:stop] += weight[:chunk_length]
            del chunk_out
        assert sum_weight.min() > 0
        out /= sum_weight # normalize
        return out
    else:
        mix = mix.unsqueeze(0) # assume batch_size=1
        # -------------------------
        # model
        # -------------------------
        with torch.no_grad():
            out = model(mix.to(device)).detach().cpu()

        out = out[0]  # assume batch_size=1
        # Padding, so output will be the same size as input
        out = F.pad(out, (0, mix.size(-1) - out.size(-1)))
        return out

