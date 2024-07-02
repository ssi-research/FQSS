"""
 This file is copied from https://github.com/asteroid-team/asteroid/blob/master/asteroid/engine/system.py
 and modified for this project needs.
"""

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from asteroid.utils.generic_utils import flatten_dict
from train_env.asteroid_librimix.wsdr import *
from asteroid.losses import *
from process import quantize

EPS=1e-8

def split_msb_lsb(x, n_bits=8, sign=True):
    x = x[0]
    threshold = max(abs(x.min()), abs(x.max()))
    x_msb = quantize(x, threshold=threshold, n_bits=n_bits, sign=sign)
    delta = 1 / (2 ** (n_bits - int(sign)))
    x_lsb = (x - x_msb)/(0.5 * delta)
    return x_msb, x_lsb

class System(pl.LightningModule):
    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Note that by default, any PyTorch-Lightning hooks are *not* passed to the model.
    If you want to use Lightning hooks, add the hooks to a subclass::

        class MySystem(System):
            def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
                return self.model.on_train_batch_start(batch, batch_idx, dataloader_idx)

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.

    .. note:: By default, ``training_step`` (used by ``pytorch-lightning`` in the
        training loop) and ``validation_step`` (used for the validation loop)
        share ``common_step``. If you want different behavior for the training
        loop and the validation loop, overwrite both ``training_step`` and
        ``validation_step`` instead.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model,
        fmodel,
        kd_lambda,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.fmodel = fmodel if kd_lambda>0 else None
        self.kd_lambda = kd_lambda
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.kd_sisdr_func = PITLossWrapper(pairwise_wsisdr, pit_from="pw_mtx")
        self.config = {} if config is None else config
        # Save lightning's AttributeDict under self.hparams
        self.save_hyperparameters(self.config_to_hparams(self.config))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step_msb(self, batch, batch_nb, train=True):
        inputs, targets = batch
        est_targets, dec_out = self(inputs)
        # --------------------
        # Loss
        # --------------------
        if train and self.kd_lambda > 0:
            sdrs, sdrqs = [], []
            with torch.no_grad():
                fest_targets, fdec_out = self.fmodel(inputs)
                fest_targets = fest_targets.detach()
                fdec_out = fdec_out.detach()
                for idx in range(len(fest_targets)):
                    sdr = self.loss_func(fest_targets[idx:idx+1], targets[idx:idx+1]).detach()
                    sdrs.append(sdr)
                    sdrq = self.loss_func(est_targets[idx:idx+1], targets[idx:idx+1]).detach()
                    sdrqs.append(sdrq)
                sdrs = torch.stack(sdrs)
                sdrqs = torch.stack(sdrqs)
                w = 10**((sdrs - sdrqs) / 10)

            fdec_out_msb, fdec_out_lsb = split_msb_lsb(fdec_out)
            kd_sdr_msb = self.kd_sisdr_func(dec_out[0].squeeze(1), fdec_out_msb.squeeze(1), weights=w)
            kd_sdr_lsb = self.kd_sisdr_func(dec_out[1].squeeze(1), fdec_out_lsb.squeeze(1), weights=w)
            task_sdr = self.kd_sisdr_func(est_targets, targets)
            loss = -10*torch.log10((1-self.kd_lambda)*task_sdr + 0.5*self.kd_lambda*kd_sdr_lsb + 0.5*self.kd_lambda*kd_sdr_msb + EPS)
            return loss, -10*torch.log10(0.5*kd_sdr_msb+0.5*kd_sdr_lsb+EPS)

        else:
            # Simple loss
            loss = self.loss_func(est_targets, targets)
            return loss, 0

    def common_step(self, batch, batch_nb, train=True):
        inputs, targets = batch
        est_targets = self(inputs)
        # --------------------
        # Loss
        # --------------------
        if train and self.kd_lambda > 0:
            sdrs, sdrqs = [], []
            with torch.no_grad():
                fest_targets = self.fmodel(inputs).detach()
                for idx in range(len(fest_targets)):
                    sdr = self.loss_func(fest_targets[idx:idx+1], targets[idx:idx+1]).detach()
                    sdrs.append(sdr)
                    sdrq = self.loss_func(est_targets[idx:idx+1], targets[idx:idx+1]).detach()
                    sdrqs.append(sdrq)
                sdrs = torch.stack(sdrs)
                sdrqs = torch.stack(sdrqs)
                w = 10**((sdrs - sdrqs) / 10)

            kd_sdr = -self.kd_sisdr_func(est_targets, fest_targets, weights=w)
            task_sdr = -self.kd_sisdr_func(est_targets, targets)
            loss = -10*torch.log10((1-self.kd_lambda)*task_sdr + self.kd_lambda*kd_sdr + EPS)
            return loss, -10*torch.log10(kd_sdr+EPS)

        else:
            # Simple loss
            loss = self.loss_func(est_targets, targets)
            return loss, 0

    def training_step(self, batch, batch_nb):
        loss, kd_loss = self.common_step(batch, batch_nb, train=True)
        self.log("loss", loss, logger=True)
        self.log("kd_loss", kd_loss, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, _ = self.common_step(batch, batch_nb, train=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
