"""
 This file is copied from: https://github.com/facebookresearch/demucs/blob/v2/demucs/utils.py
 and modified for this project needs.
"""

import errno
import random
import socket
from process import preprocess, postprocess
import torch as th
import tqdm
from torch import distributed
from torch.nn import functional as F


def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


def average_metric(metric, count=1.):
    """
    Average `metric` which should be a float across all hosts. `count` should be
    the weight for this particular host (i.e. number of examples).
    """
    metric = th.tensor([count, count * metric], dtype=th.float32, device='cuda')
    distributed.all_reduce(metric, op=distributed.ReduceOp.SUM)
    return metric[1].item() / metric[0].item()


def free_port(host='', low=20000, high=40000):
    """
    Return a port number that is most likely free.
    This could suffer from a race condition although
    it should be quite rare.
    """
    sock = socket.socket()
    while True:
        port = random.randint(low, high)
        try:
            sock.bind((host, port))
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                continue
            raise
        return port


def human_seconds(seconds, display='.2f'):
    value = seconds * 1e6
    ratios = [1e3, 1e3, 60, 60, 24]
    names = ['us', 'ms', 's', 'min', 'hrs', 'days']
    last = names.pop(0)
    for name, ratio in zip(names, ratios):
        if value / ratio < 0.3:
            break
        value /= ratio
        last = name
    return f"{format(value, display)} {last}"


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        self.tensor = tensor
        self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, th.Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model(model, mix, split=False, segment=4*80000, sample_rate=44100,
                overlap=0.25, progress=False):
    """
    Apply model to a given mixture
    """
    device = mix.device
    channels, length = mix.shape
    if split:
        out = th.zeros(len(model.sources), channels, length, device=device)
        sum_weight = th.zeros(length, device=device)
        stride = int((1 - overlap) * segment)
        offsets = range(0, length, stride)
        scale = stride / sample_rate
        if progress:
            offsets = tqdm.tqdm(offsets, unit_scale=scale, ncols=120, unit='seconds')
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = th.cat([th.arange(1, segment // 2 + 1), th.arange(segment - segment // 2, 0, -1)]).to(device)
        assert len(weight) == segment
        weight = weight / weight.max()
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment)
            chunk_out = apply_model(model, chunk)
            chunk_length = chunk_out.shape[-1]
            out[..., offset:offset + segment] += weight[:chunk_length] * chunk_out
            sum_weight[offset:offset + segment] += weight[:chunk_length]
            offset += segment
        assert sum_weight.min() > 0
        out /= sum_weight
        return out
    else:
        mix = tensor_chunk(mix)
        padded_mix = mix.padded(length)
        with th.no_grad():
            padded_mix = padded_mix.unsqueeze(0)  # assume batch_size=1
            # ------------------------
            # Model
            # ------------------------
            out = model(padded_mix)
            # ------------------------
            out = out[0]  # assume batch_size=1
            # Padding, so output will be the same size as input
            out = F.pad(out, (0, padded_mix.size(-1) - out.size(-1)))
        return center_trim(out, length)

