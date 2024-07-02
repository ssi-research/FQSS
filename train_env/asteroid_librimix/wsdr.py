"""
 This file is copied from https://github.com/asteroid-team/asteroid/blob/master/asteroid/losses/sdr.py
 and modified for this project needs.
"""

import torch
from torch.nn.modules.loss import _Loss


class SDR(_Loss):

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super(SDR, self).__init__()
        assert sdr_type in ["sisdr", "sdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets, weights=None):
        assert targets.size() == est_targets.size()
        # Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=-1, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=-1, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate

        pair_wise_dot = torch.sum(est_targets * targets, dim=-1, keepdim=True)
        s_target_energy = torch.sum(targets**2, dim=-1, keepdim=True) + self.EPS
        pair_wise_proj = pair_wise_dot * targets / s_target_energy
        if self.sdr_type in ["sisdr"]:
            noise = est_targets - pair_wise_proj
        else:
            noise = est_targets - targets
        pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=-1) / (torch.sum(noise**2, dim=-1) + self.EPS)
        if weights is not None:
            pair_wise_sdr = pair_wise_sdr * weights[:, None, None]
        pair_wise_sdr = torch.mean(pair_wise_sdr)
        if self.take_log:
            return 10 * torch.log10(pair_wise_sdr + self.EPS)
        else:
            return pair_wise_sdr


class PairwiseWSDR(_Loss):

    def __init__(self, sdr_type, zero_mean=True, take_log=True, EPS=1e-8):
        super(PairwiseWSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS

    def forward(self, est_targets, targets, weights=None):
        if targets.size() != est_targets.size() or targets.ndim != 3:
            raise TypeError(
                f"Inputs must be of shape [batch, n_src, time], got {targets.size()} and {est_targets.size()} instead"
            )
        assert targets.size() == est_targets.size()
        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # Step 2. Pair-wise SI-SDR. (Reshape to use broadcast)
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)

        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, n_src, n_src, 1]
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            # [batch, 1, n_src, 1]
            s_target_energy = torch.sum(s_target**2, dim=3, keepdim=True) + self.EPS
            # [batch, n_src, n_src, time]
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            # [batch, n_src, n_src, time]
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=3) / (
            torch.sum(e_noise**2, dim=3) + self.EPS
        )
        if weights is not None:
            pair_wise_sdr = pair_wise_sdr * weights[:, None, None]
        if self.take_log:
            return 10 * torch.log10(pair_wise_sdr + self.EPS)
        else:
            return -pair_wise_sdr


# aliases
sisdr = SDR("sisdr", take_log=False)
sdr = SDR("sdr", take_log=False)
pairwise_wsisdr = PairwiseWSDR("sisdr", take_log=False)
pairwise_wsdsdr = PairwiseWSDR("sdsdr", take_log=False)