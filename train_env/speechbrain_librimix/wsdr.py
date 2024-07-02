"""
 This file is copied from: https://github.com/speechbrain/speechbrain/blob/main/speechbrain/nnet/losses.py
 and modified for this project needs.
"""


import torch
from speechbrain.nnet.losses import get_mask
from itertools import permutations

EPS = 1e-8

def cal_w_si_snr(source, estimate_source, weights=None):

    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[-2], device=device
    )
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
        torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = si_snr_beforelog.unsqueeze(0)
    if weights is None:
        return -si_snr
    else:
        return -si_snr*weights.unsqueeze(0)


class PitWrapper(torch.nn.Module):

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):

        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target, weights):
        n_sources = pred.size(-1)

        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )

        loss_mat = self.base_loss(pred, target, weights)
        assert (
            len(loss_mat.shape) >= 2
        ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)

    def reorder_tensor(self, tensor, p):
        reordered = torch.zeros_like(tensor, device=tensor.device)
        for b in range(tensor.shape[0]):
            reordered[b] = tensor[b][..., p[b]].clone()
        return reordered

    def forward(self, preds, targets, weights=None):
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label, weights)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms


def get_w_si_snr_with_pitwrapper(source, estimate_source, weights=None):
    pit_si_snr = PitWrapper(cal_w_si_snr)
    loss, perms = pit_si_snr(source, estimate_source, weights)
    return loss
