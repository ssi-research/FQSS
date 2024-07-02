"""
 This file is copied from: https://github.com/speechbrain/speechbrain/blob/main/recipes/LibriMix/separation/train.py
 and modified for this project needs.
"""

#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on Libri2/3Mix datasets.
The system employs an encoder, a decoder, and a masking network.
"""
import copy

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from speechbrain.utils.distributed import run_on_main
from torch.cuda.amp import autocast
from hyperpyyaml import load_hyperpyyaml
import numpy as np
from tqdm import tqdm
import csv
import logging
from utils import set_seed
from train_env.speechbrain_librimix.prepare_data import prepare_librimix

from quantization.qat.models.load_model import quantize_model

# Logger info
logger = logging.getLogger(__name__)

EPS = 1e-8

# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [targets[i][0].unsqueeze(-1) for i in range(self.hparams.num_spks)],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    mix = targets.sum(-1)

                    if self.hparams.use_wham_noise:
                        noise = noise.to(self.device)
                        len_noise = noise.shape[1]
                        len_mix = mix.shape[1]
                        min_len = min(len_noise, len_mix)

                        # add the noise
                        mix = mix[:, :min_len] + noise[:, :min_len]

                        # fix the length of targets also
                        targets = targets[:, :min_len, :]

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        est_source = self.hparams.Sepformer(mix)
        T_origin = mix.size(-1)
        T_est = est_source.size(-1)
        est_source_pp = est_source.transpose(-2,-1)
        if T_origin > T_est:
            est_source_pp = F.pad(est_source_pp, (0, 0, 0, T_origin - T_est))
        else:
            est_source_pp = est_source_pp[:, :T_origin, :]

        if stage == sb.Stage.TRAIN and self.hparams.kd_lambda > 0:
            with torch.no_grad():
                fest_source = self.modules.fmodel(mix).detach()
                fest_source_pp = fest_source.transpose(-2,-1)
                if T_origin > T_est:
                    fest_source_pp = F.pad(fest_source_pp, (0, 0, 0, T_origin - T_est))
                else:
                    fest_source_pp = fest_source_pp[:, :T_origin, :]
            return est_source_pp, targets, fest_source_pp

        return est_source_pp, targets

    def compute_kd_objectives(self, predictions, targets, fpredictions):
        sdrs, sdrqs = [], []
        with torch.no_grad():
            for idx in range(len(fpredictions)):
                sdr = self.hparams.loss(targets[idx:idx+1], fpredictions[idx:idx+1]).detach()
                sdrs.append(sdr)
                sdrq = self.hparams.loss(targets[idx:idx+1], predictions[idx:idx+1]).detach()
                sdrqs.append(sdrq)
            sdrs = torch.stack(sdrs)
            sdrqs = torch.stack(sdrqs)
            w = 10**((sdrs-sdrqs)/10)

        kd_lambda = self.hparams.kd_lambda
        kd_sdr = -self.hparams.loss_kd(predictions, fpredictions, weights=w)
        task_sdr = -self.hparams.loss_kd(predictions, targets)
        loss = -10*torch.log10((1-kd_lambda)*task_sdr + kd_lambda*kd_sdr + EPS)
        return loss

    def compute_objectives(self, predictions, targets):
        """Computes the si-snr loss"""
        return self.hparams.loss(targets, predictions)

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.use_wham_noise:
            noise = batch.noise_sig[0]
        else:
            noise = None

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.auto_mix_prec:
            with autocast():
                outputs = self.compute_forward(mixture, targets, sb.Stage.TRAIN, noise)
                if len(outputs)==3:
                    loss = self.compute_kd_objectives(*outputs)
                else:
                    loss = self.compute_objectives(*outputs)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if loss < self.hparams.loss_upper_lim and loss.nelement() > 0:  # the fix for computational problems

                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.modules.parameters(), self.hparams.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                #loss.data = torch.tensor(0).to(self.device)
                print("Warning: Loss is {}, skipping this sample! nonfinite_count={}".format(loss, self.nonfinite_count))
        else:
            outputs = self.compute_forward(mixture, targets, sb.Stage.TRAIN, noise)
            if len(outputs) == 3:
                loss = self.compute_kd_objectives(*outputs)
            else:
                loss = self.compute_objectives(*outputs)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            if loss < self.hparams.loss_upper_lim and loss.nelement() > 0:  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(self.modules.parameters(), self.hparams.clip_grad_norm)
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                #loss.data = torch.tensor(0).to(self.device)
                print("Warning: Loss is {}, skipping this sample! nonfinite_count={}".format(loss, self.nonfinite_count))
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, "n_audio_to_save"):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.work_dir, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts
        )

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST
                        )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                    if i % 500 == 0:
                        print("Mean SISNR is {:0.3f}".format(np.array(all_sisnrs).mean()))
                        print("Mean SISNRi is {:0.3f}".format(np.array(all_sisnrs_i).mean()))
                        print("Mean SDR is {:0.3f}".format(np.array(all_sdrs).mean()))
                        print("Mean SDRi is {:0.3f}".format(np.array(all_sdrs_i).mean()))

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        sisnr = np.array(all_sisnrs).mean()
        sisnri = np.array(all_sisnrs_i).mean()
        sdr = np.array(all_sdrs).mean()
        sdri = np.array(all_sdrs_i).mean()
        logger.info("Mean SISNR is {}".format(sisnr))
        logger.info("Mean SISNRi is {}".format(sisnri))
        logger.info("Mean SDR is {}".format(sdr))
        logger.info("Mean SDRi is {}".format(sdri))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


def dataio_prep(hparams, dataset_cfg):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=dataset_cfg["train_data"],
        replacements={"data_root": dataset_cfg["data_folder"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=dataset_cfg["valid_data"],
        replacements={"data_root": dataset_cfg["data_folder"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=dataset_cfg["test_data"],
        replacements={"data_root": dataset_cfg["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines
    new_sample_rate = int(dataset_cfg['sample_rate'] * dataset_cfg['resample'])

    @sb.utils.data_pipeline.takes("mix_wav")
    @sb.utils.data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = sb.dataio.dataio.read_audio(mix_wav)
        mix_sig = torchaudio.transforms.Resample(dataset_cfg['sample_rate'],new_sample_rate)(mix_sig)
        return mix_sig

    @sb.utils.data_pipeline.takes("s1_wav")
    @sb.utils.data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = sb.dataio.dataio.read_audio(s1_wav)
        s1_sig = torchaudio.transforms.Resample(dataset_cfg['sample_rate'],new_sample_rate)(s1_sig)
        return s1_sig

    @sb.utils.data_pipeline.takes("s2_wav")
    @sb.utils.data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = sb.dataio.dataio.read_audio(s2_wav)
        s2_sig = torchaudio.transforms.Resample(dataset_cfg['sample_rate'],new_sample_rate)(s2_sig)
        return s2_sig

    if hparams["num_spks"] == 3:

        @sb.utils.data_pipeline.takes("s3_wav")
        @sb.utils.data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = sb.dataio.dataio.read_audio(s3_wav)
            s3_sig = torchaudio.transforms.Resample(dataset_cfg['sample_rate'],new_sample_rate)(s3_sig)
            return s3_sig

    if dataset_cfg["noisy"]:

        @sb.utils.data_pipeline.takes("noise_wav")
        @sb.utils.data_pipeline.provides("noise_sig")
        def audio_pipeline_noise(noise_wav):
            noise_sig = sb.dataio.dataio.read_audio(noise_wav)
            noise_sig = torchaudio.transforms.Resample(dataset_cfg['sample_rate'],new_sample_rate)(noise_sig)
            return noise_sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_mix)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s1)
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_s3)

    if dataset_cfg["noisy"]:
        print("Using the WHAM! noise in the data pipeline")
        sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline_noise)

    if (hparams["num_spks"] == 2) and dataset_cfg["noisy"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
        )
    elif (hparams["num_spks"] == 3) and dataset_cfg["noisy"]:
        sb.dataio.dataset.set_output_keys(
            datasets,
            ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"],
        )
    elif (hparams["num_spks"] == 2) and not dataset_cfg["noisy"]:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig"]
        )
    else:
        sb.dataio.dataset.set_output_keys(
            datasets, ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
        )

    return train_data, valid_data, test_data

def train(yml_path, local_rank, distributed_launch, device):

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments([yml_path])
    run_opts.update({"local_rank":local_rank, "distributed_launch":distributed_launch})

    with open(yml_path) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    model_cfg = hparams["model_cfg"]
    dataset_cfg = hparams["dataset_cfg"]

    # Ensuring training reproducibility
    seed = hparams.get("seed", 0)
    set_seed(seed)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["work_dir"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_librimix,
        kwargs={
            "datapath": dataset_cfg["data_folder"],
            "savepath": hparams["save_folder"],
            "n_spks": hparams["num_spks"],
            "skip_prep": dataset_cfg["skip_prep"],
            "librimix_addnoise": hparams["use_wham_noise"],
        },
    )

    dataset_cfg["train_data"] = os.path.join(hparams["save_folder"], "libri2mix_train-360.csv")
    dataset_cfg["valid_data"] = os.path.join(hparams["save_folder"], "libri2mix_dev.csv")
    dataset_cfg["test_data"] = os.path.join(hparams["save_folder"], "libri2mix_test.csv")
    train_data, valid_data, test_data = dataio_prep(hparams, dataset_cfg)

    # Load pretrained model if pretrained_separator is present in the yaml
    if "pretrained_separator" in hparams:
        run_on_main(hparams["pretrained_separator"].collect_files)
        hparams["pretrained_separator"].load_collected()

    # ---------------------------------
    # Load model
    # ---------------------------------
    model = hparams['modules']["model"]
    if hparams.get("kd_lambda", 0) > 0:
        fmodel = copy.deepcopy(model)
        fmodel.to(device)
        fmodel.eval()
        hparams['modules']["fmodel"] = fmodel


    model = quantize_model(model, model_cfg['quantization'])
    model.to(device)
    model.train()

    # ---------------------------------
    # Brain class initialization
    # ---------------------------------
    if device == "cpu":
        run_opts.update({"device":device})

    separator = Separation(
        modules=hparams['modules'],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # re-initialize the parameters if we don't use a pretrained model
    if "pretrained_separator" not in hparams:
        for module in separator.modules.values():
            separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        # Training
        separator.fit(
            separator.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )

    # Eval
    separator.evaluate(test_data, min_key="si-snr")
