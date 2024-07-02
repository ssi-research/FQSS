"""
 This file is copied from: https://github.com/facebookresearch/demucs/blob/v2/demucs/train.py
 and modified for this project needs.
"""

import json
import os
import sys
import time
import yaml
from dataclasses import dataclass, field
import torch
from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
from musdbhq_dataset import get_musdb_wav_datasets
from demucs.augment import FlipChannels, FlipSign, Remix, Scale, Shift
from utils import set_seed
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from musdbhq_utils import apply_model, average_metric, center_trim, human_seconds
import argparse
from pathlib import Path
from train_env.train_utils import create_pretrained_model
from process import calc_nsdr

@dataclass
class SavedState:
    metrics: list = field(default_factory=list)
    last_state: dict = None
    best_state: dict = None
    optimizer: dict = None


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yml_path", '-y', type=str, required=True, help="YML configuration file")
    parser.add_argument("--device", "-d", help="Device to train on: cpu or cuda")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master")
    return parser


def train_model(epoch, dataset, model, fmodel,
                loss_fn, optimizer, augment,
                device="cpu", seed=None, kd_lambda=0, workers=4,
                world_size=1, batch_size=8):

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset)
        sampler_epoch = epoch
        if seed is not None:
            sampler_epoch += seed * 1000
        sampler.set_epoch(sampler_epoch)
        batch_size //= world_size
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    tq = tqdm.tqdm(loader,
                   ncols=120,
                   desc=f"[{epoch:03d}]",
                   leave=False,
                   file=sys.stdout,
                   unit=" batch")

    total_loss, current_loss = 0, 0
    for idx, sources in enumerate(tq):
        if len(sources) < batch_size:
            continue  # skip uncomplete batch for augment.Remix to work properly

        # Create mix
        sources = sources.to(device)
        sources = augment(sources)
        mix = sources.sum(dim=1)

        # ------------------------
        # Model
        # ------------------------
        wavs = model(mix)
        sources = center_trim(sources, wavs)

        # Loss
        # ------------------------
        if kd_lambda > 0:
            sdrs, sdrqs = [], []
            with torch.no_grad():
                fwavs = fmodel(mix).detach()
                for i in range(len(fwavs)):
                    sdr = calc_nsdr(fwavs[i:i+1], sources[i:i+1])
                    sdrs.append(sdr)
                    sdrq = calc_nsdr(wavs[i:i+1], sources[i:i+1])
                    sdrqs.append(sdrq)
                sdrs = torch.Tensor(sdrs)
                sdrqs = torch.Tensor(sdrqs)
                w = 10**((sdrs - sdrqs) / 10)

            kd_losses = []
            for i in range(len(fwavs)):
                kd_loss_i = loss_fn(wavs[i:i+1], fwavs[i:i+1])
                kd_losses.append(kd_loss_i)
            kd_loss = torch.stack(kd_losses, dim=0)
            kd_loss = torch.mean(w.to(device) * kd_loss)
            task_loss = loss_fn(wavs, sources)
            loss = (1-kd_lambda)*task_loss + kd_lambda*kd_loss
        else:
            loss = loss_fn(wavs, sources)

        # Back-Prop
        loss.backward()

        # Calc grad norm
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm()**2
        grad_norm = grad_norm**0.5

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        current_loss = total_loss / (1 + idx)
        tq.set_postfix(loss=f"{current_loss:.4f}", grad=f"{grad_norm:.5f}")

        # Free some space before next round
        del sources, mix, wavs, loss


    if world_size > 1:
        sampler.epoch += 1
        current_loss = average_metric(current_loss)

    return current_loss


def validate_model(epoch, dataset, model,
                   loss_fn, device="cpu",
                   rank=0, world_size=1,
                   overlap=0.25, split=False):
    indexes = range(rank, len(dataset), world_size)
    tq = tqdm.tqdm(indexes,
                   ncols=120,
                   desc=f"[{epoch:03d}] valid",
                   leave=False,
                   file=sys.stdout,
                   unit=" track")
    current_loss = 0
    for index in tq:
        streams = dataset[index]
        streams = streams[..., :15_000_000] # first five minutes to avoid OOM on --upsample models
        streams = streams.to(device)
        sources = streams[1:]
        mix = streams[0]
        estimates = apply_model(model, mix,
                                split=split,
                                overlap=overlap)
        loss = loss_fn(estimates, sources)
        current_loss += loss.item() / len(indexes)
        del estimates, streams, sources

    if world_size > 1:
        current_loss = average_metric(current_loss, len(indexes))
    return current_loss


def main():

    parser = get_parser()
    args = parser.parse_args()
    device = args.device

    # ------------------------------------------
    # Read yml
    # ------------------------------------------
    with open(args.yml_path) as f:
        conf = yaml.safe_load(f)

    work_dir, model_cfg, dataset_cfg = conf['work_dir'], conf['model_cfg'], conf['dataset_cfg']
    training_cfg, testing_cfg = conf['training_cfg'], conf['testing_cfg']

    # Working path
    test_name = work_dir.split('/')[-1]
    work_dir = Path(work_dir)
    work_dir.mkdir(exist_ok=True)
    metrics_path = work_dir / f"{test_name}.json"
    checkpoint = work_dir / "checkpoint.pth"
    checkpoint_tmp = work_dir / "checkpoint.pth.tmp"
    best_model_path = work_dir / "best_model.pth"
    latest_model_path = work_dir / "latest_model.pth"

    # Ensuring training reproducibility
    seed = training_cfg.get("seed", 0)
    set_seed(seed)

    # Prevents too many threads to be started when running `museval` as it can be quite inefficient on NUMA architectures.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # ------------------------------------------
    # Multi GPUs
    # ------------------------------------------
    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print("Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

    # ------------------------------------------
    # Load model
    # ------------------------------------------
    model_cfg.update({"model_path":training_cfg['pretrained']})
    model, fmodel = create_pretrained_model(model_cfg)
    model.to(device)
    model.train()
    fmodel.to(device)
    fmodel.eval()

    # ------------------------------------------
    # Save configuration
    # ------------------------------------------
    conf_path = os.path.join(work_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(model_cfg, outfile)
        yaml.safe_dump(dataset_cfg, outfile)
        yaml.safe_dump(training_cfg, outfile)

    # WandB
    enable_wandb = training_cfg.get("wandb", False)
    if enable_wandb:
        import wandb
        print("WandB is enable!")
        PROJECT_NAME = "ConvTasNet_mus_sep"
        wandb.init(project=PROJECT_NAME, group=test_name, dir=work_dir)

    # ------------------------------------------
    # Training setup
    # ------------------------------------------
    epochs = training_cfg.get("epochs", 20)
    lr = training_cfg["optim"].get("lr", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # Resume training
    try:
        saved = torch.load(checkpoint, map_location='cpu')
    except IOError:
        saved = SavedState()
    if saved.last_state is not None:
        model.load_state_dict(saved.last_state, strict=False)
    if saved.optimizer is not None:
        optimizer.load_state_dict(saved.optimizer)


    # ------------------------------------------
    # Dataset
    # ------------------------------------------
    musdb_train_dir = dataset_cfg.get("train_dir", 44100)
    data_stride = dataset_cfg.get("data_stride", 44100)
    sample_rate = dataset_cfg.get("sample_rate", 44100)
    segment_samples = dataset_cfg.get("segment_samples", 80000)
    print(f"Number of training samples adjusted to {segment_samples}")
    samples = segment_samples + data_stride
    train_set, valid_set = get_musdb_wav_datasets(musdb_train_dir, data_stride, sample_rate,
                                                  samples, model.sources,
                                                  dataset_cfg["metadata"], args.world_size)
    print("Train set and valid set sizes", len(train_set), len(valid_set))

    # ------------------------------------------
    # Augmentation
    # ------------------------------------------
    batch_size = training_cfg.get("batch_size", 4)
    augment = [Shift(data_stride), FlipSign(), FlipChannels(), Scale()] #, Remix(max(1,batch_size//4))]
    augment = nn.Sequential(*augment).to(device)
    print("Agumentation pipeline:", augment)

    best_loss = float("inf")
    for epoch, metrics in enumerate(saved.metrics):
        print(f"Epoch {epoch:03d}: "
              f"train={metrics['train']:.8f} "
              f"valid={metrics['valid']:.8f} "
              f"best={metrics['best']:.4f} "
              f"duration={human_seconds(metrics['duration'])}")
        best_loss = metrics['best']


    dmodel = model
    if args.world_size > 1:
        dmodel = DistributedDataParallel(model,
                                         device_ids=[torch.cuda.current_device()],
                                         output_device=torch.cuda.current_device(),
                                         find_unused_parameters=True)

    # ----------------------------
    # Training Validation loop
    # ----------------------------
    num_workers = training_cfg.get("num_workers",4)
    overlap = testing_cfg.get("overlap", 0.25)
    kd_lambda = training_cfg.get("kd_lambda",0)
    for epoch in range(len(saved.metrics), epochs):
        begin = time.time()

        # Training
        # ---------
        model.train()
        train_loss = train_model(epoch, train_set, dmodel, fmodel, loss_fn, optimizer, augment,
                                batch_size=batch_size,
                                device=device,
                                kd_lambda=kd_lambda,
                                seed=seed,
                                workers=num_workers,
                                world_size=args.world_size)

        # Validation
        # -----------
        model.eval()
        valid_loss = validate_model(epoch, valid_set, model, loss_fn,
                                    device=device,
                                    rank=args.rank,
                                    split=True,
                                    overlap=overlap,
                                    world_size=args.world_size)

        duration = time.time() - begin

        if valid_loss < best_loss:
            best_loss = valid_loss
            saved.best_state = {key: value.to("cpu").clone() for key, value in model.state_dict().items()}
            # Save model
            if args.rank == 0:
                torch.save(model.state_dict(), best_model_path)

        if args.rank == 0:
            torch.save(model.state_dict(), latest_model_path)

        saved.metrics.append({
            "train": train_loss,
            "valid": valid_loss,
            "best": best_loss,
            "duration": duration,
        })

        if args.rank == 0:
            json.dump(saved.metrics, open(metrics_path, "w"))

        if enable_wandb:
            wandb.log({"train": train_loss, "valid": valid_loss, "best": best_loss})

        saved.last_state = model.state_dict()
        saved.optimizer = optimizer.state_dict()
        if args.rank == 0:
            torch.save(saved, checkpoint_tmp)
            checkpoint_tmp.rename(checkpoint)

        print(f"Epoch {epoch:03d}: train={train_loss:.8f}, valid={valid_loss:.8f}, best={best_loss:.4f}, duration={human_seconds(duration)}")


if __name__ == "__main__":
    main()
