import yaml
import os
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from train_env.asteroid_librimix.librimix_dataset import LibriMix

from train_env.asteroid_librimix.mysystem import System
from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import *

from utils import set_seed
from train_env.train_utils import create_pretrained_model


def prepare_datasets(dataset_cfg, training_cfg):
    # Augmentation
    augmentation_cfg = dataset_cfg.get('augmentation',None)
    if augmentation_cfg:
        enable = augmentation_cfg.get("enable", False)
        if not enable:
            augmentation_cfg = None

    # Train dataset
    train_set = LibriMix(
        csv_dir=dataset_cfg["train_dir"],
        task=dataset_cfg["task"],
        sample_rate=dataset_cfg["sample_rate"],
        resample=dataset_cfg.get("resample",1),
        n_src=dataset_cfg["n_src"],
        segment=dataset_cfg["segment"],
        augmentation_cfg=augmentation_cfg,
    )

    # Validation dataset
    val_set = LibriMix(
        csv_dir=dataset_cfg["valid_dir"],
        task=dataset_cfg["task"],
        sample_rate=dataset_cfg["sample_rate"],
        resample=dataset_cfg.get("resample",1),
        n_src=dataset_cfg["n_src"],
        segment=dataset_cfg["segment"],
    )

    print("Training set size: {}".format(len(train_set)))
    print("Validation set size: {}".format(len(val_set)))

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        drop_last=True,
    )

    return train_loader, val_loader


def generate_model(model_cfg, training_cfg, device):
    pretrained_path = training_cfg.get("pretrained", None)
    model_cfg.update({"model_path":pretrained_path})
    is_ckpt = False
    model, fmodel = create_pretrained_model(model_cfg)
    if pretrained_path is not None:
        if pretrained_path.endswith(".ckpt"):
            is_ckpt = True
    # Quantized model
    model.to(device)
    model.train()
    # Float model
    fmodel.to(device)
    fmodel.eval()
    return model, fmodel, pretrained_path, is_ckpt



def train_setup(model, fmodel, train_loader, val_loader, training_cfg, work_dir, device, wandbLogger=None):
    # ------------------------------------
    # Training Setup
    # ------------------------------------
    optimizer = make_optimizer(model.parameters(), **training_cfg["optim"])
    # Define scheduler
    scheduler = None
    if training_cfg.get("half_lr",False):
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=training_cfg.get("patience",5))
    elif training_cfg.get("step_lr",None) is not None:
        step_lr = training_cfg["step_lr"]
        scheduler = StepLR(optimizer=optimizer, step_size=step_lr.get("step_size",2), gamma=step_lr.get("gamma",0.98))
    kd_lambda = training_cfg.get("kd_lambda",0)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    system = System(
        model=model,
        fmodel=fmodel,
        kd_lambda=kd_lambda,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(work_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor="val_loss", mode="min", save_top_k=3, verbose=True)
    callbacks.append(checkpoint)
    if training_cfg["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))

    trainer = pl.Trainer(
        max_epochs=training_cfg["epochs"],
        callbacks=callbacks,
        default_root_dir=work_dir,
        accelerator="gpu" if torch.cuda.is_available() and device!='cpu' else "cpu",
        strategy="ddp",
        devices="auto",
        gradient_clip_val=5.0,
        logger=wandbLogger,
        gpus=training_cfg.get("gpus",None),
    )

    return trainer, system, checkpoint


def train(yml_path, device):

    # -----------------------------------
    # Read yml
    # -----------------------------------
    with open(yml_path) as f:
        conf = yaml.safe_load(f)

    work_dir, model_cfg, dataset_cfg = conf['work_dir'], conf['model_cfg'], conf['dataset_cfg']
    training_cfg, testing_cfg = conf['training_cfg'], conf['testing_cfg']

    # Ensuring training reproducibility
    seed = training_cfg.get("seed", 0)
    set_seed(seed)

    # ------------------------------------
    # Dataset
    # ------------------------------------
    train_loader, val_loader = prepare_datasets(dataset_cfg, training_cfg)
    
    # ------------------------------------
    # Model
    # ------------------------------------
    model, fmodel, pretrained_path, is_ckpt = generate_model(model_cfg, training_cfg, device)

    # Just after instantiating, save the args. Easy loading in the future.
    os.makedirs(work_dir, exist_ok=True)
    conf_path = os.path.join(work_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(model_cfg, outfile)
        yaml.safe_dump(dataset_cfg, outfile)
        yaml.safe_dump(training_cfg, outfile)

    # ------------------------------------
    # WandB
    # ------------------------------------
    wandbLogger = None
    if training_cfg.get("wandb", False):
        import wandb
        print("WandB is enable!")
        test_name = work_dir.split('/')[-1]
        PROJECT_NAME = model_cfg["name"] + "_" + dataset_cfg["task"]
        wandb.init(project=PROJECT_NAME, group=test_name, dir=work_dir)
        wandbLogger = WandbLogger(project=PROJECT_NAME, group=test_name, dir=work_dir)

    # ------------------------------------
    # Training setup
    # ------------------------------------
    trainer, system, checkpoint = train_setup(model, fmodel, train_loader, val_loader, training_cfg, work_dir, device, wandbLogger)

    # ------------------------------------
    # Training
    # ------------------------------------
    trainer.fit(system, ckpt_path=pretrained_path if is_ckpt else None)


    # ------------------------------------
    # Post Training
    # ------------------------------------
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(work_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)


    # Save latest model
    system.cpu()
    latest_path = os.path.join(work_dir, "latest_model.pth")
    torch.save(system.model.state_dict(), latest_path)

    # Save best model
    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()
    best_path = os.path.join(work_dir, "best_model.pth")
    torch.save(system.model.state_dict(), best_path)