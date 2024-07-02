"""
 This file is copied from: https://github.com/facebookresearch/demucs/blob/v2/demucs/augment.py
 and modified for this project needs.
"""

from collections import OrderedDict
import math
import json
from pathlib import Path
import torch as th
from torch import distributed
import torchaudio as ta
from torch.nn import functional as F
import musdb
from torch import nn
import random

MIXTURE = "mixture"
EXT = ".wav"

class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.
    """
    def __init__(self, shift=8192):
        super().__init__()
        self.shift = shift

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                offsets = th.randint(self.shift, [batch, sources, 1, 1], device=wav.device)
                offsets = offsets.expand(-1, -1, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class FlipChannels(nn.Module):
    """
    Flip left-right channels.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training and wav.size(2) == 2:
            left = th.randint(2, (batch, sources, 1, 1), device=wav.device)
            left = left.expand(-1, -1, -1, time)
            right = 1 - left
            wav = th.cat([wav.gather(2, left), wav.gather(2, right)], dim=2)
        return wav


class FlipSign(nn.Module):
    """
    Random sign flip.
    """
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = th.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """
    Shuffle sources to make new mixes.
    """
    def __init__(self, group_size=4):
        """
        Shuffle sources within one batch.
        Each batch is divided into groups of size `group_size` and shuffling is done within
        each group separatly. This allow to keep the same probability distribution no matter
        the number of GPUs. Without this grouping, using more GPUs would lead to a higher
        probability of keeping two sources from the same track together which can impact
        performance.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device

        if self.training:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size
            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(th.rand(groups, group_size, streams, 1, 1, device=device),
                                      dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., min_val=0.25, max_val=1.25):
        super().__init__()
        self.proba = proba
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(self.min_val, self.max_val)
            wav *= scales
        return wav


class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            length=None, stride=None, normalize=True,
            sample_rate=44100):
        """
        Waveset (or mp3 set for that matter). Can be used to train
        with arbitrary sources. Each track should be one folder inside of `path`.
        The folder should contain files named `{source}.{ext}`.
        Files will be grouped according to `sources` (each source is a list of
        filenames).

        Sample rate and channels will be converted on the fly.

        `length` is the sample size to extract (in samples, not duration).
        `stride` is how many samples to move by between each example.
        """
        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.length = length
        self.stride = stride or length
        self.normalize = normalize
        self.sources = sources
        self.sample_rate = sample_rate
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_length = int(self.sample_rate * meta['length'] / meta['samplerate'])
            if length is None or track_length < length:
                examples = 1
            else:
                examples = int(math.ceil((track_length - self.length) / self.stride) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{EXT}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.length is not None:
                offset = int(math.ceil(
                    meta['samplerate'] * self.stride * index / self.sample_rate))
                num_frames = int(math.ceil(
                    meta['samplerate'] * self.length / self.sample_rate))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                wavs.append(wav)

            example = th.stack(wavs)
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.length:
                example = example[..., :self.length]
                example = F.pad(example, (0, self.length - example.shape[-1]))
            return example


def get_musdb_tracks(root, *args, **kwargs):
    mus = musdb.DB(root, *args, **kwargs)
    return {track.name: track.path for track in mus}


def get_musdb_wav_datasets(musdb, data_stride, sample_rate, samples, sources, metadata_file, world_size):
    root = musdb + "/train"

    if world_size > 1:
        distributed.barrier()
    metadata = json.load(open(metadata_file))

    train_tracks = get_musdb_tracks(musdb, is_wav=True, subsets=["train"], split="train")
    metadata_train = {name: meta for name, meta in metadata.items() if name in train_tracks}
    metadata_valid = {name: meta for name, meta in metadata.items() if name not in train_tracks}
    train_set = Wavset(root, metadata_train, sources,
                       length=samples, stride=data_stride,
                       sample_rate=sample_rate)
    valid_set = Wavset(root, metadata_valid, [MIXTURE] + sources,
                       sample_rate=sample_rate)
    return train_set, valid_set
