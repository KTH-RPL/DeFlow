"""
# Created: 2023-07-12 19:30
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow) and 
# SeFlow (https://github.com/KTH-RPL/SeFlow) projects.
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Train Model
"""

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint
)

from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

from scripts.network.dataloader import HDF5Dataset, collate_fn_pad
from scripts.pl_model import ModelWrapper

def precheck_cfg_valid(cfg):
    if cfg.loss_fn == 'seflowLoss' and cfg.add_seloss is None:
        raise ValueError("Please specify the self-supervised loss items for seflowLoss.")
    if (cfg.point_cloud_range[3] - cfg.point_cloud_range[0]) % cfg.voxel_size[0] != 0 or \
       (cfg.point_cloud_range[4] - cfg.point_cloud_range[1]) % cfg.voxel_size[1] != 0 or \
       (cfg.point_cloud_range[5] - cfg.point_cloud_range[2]) % cfg.voxel_size[2] != 0:
        # For example: 51.2/0.2=256 good, 51.2/0.3=170.67 wrong.
        raise ValueError("The voxel size should be able to divide the point cloud range to a INT.")
    return cfg

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    precheck_cfg_valid(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    train_dataset = HDF5Dataset(cfg.train_data, n_frames=cfg.num_frames, dufo=(cfg.loss_fn == 'seflowLoss'))
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_fn_pad,
                              pin_memory=True)
    val_loader = DataLoader(HDF5Dataset(cfg.val_data, n_frames=cfg.num_frames),
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            collate_fn=collate_fn_pad,
                            pin_memory=True)
                            
    # count gpus, overwrite gpus
    cfg.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    output_dir = HydraConfig.get().runtime.output_dir
    # overwrite logging folder name for SSL.
    if cfg.loss_fn == 'seflowLoss':
        cfg.output = cfg.output.replace(cfg.model.name, "seflow")
        output_dir = output_dir.replace(cfg.model.name, "seflow")
        method_name = "seflow"
    else:
        method_name = cfg.model.name

    # FIXME: hydra output_dir with ddp run will mkdir in the parent folder. Looks like PL and Hydra trying to fix in lib.
    # print(f"Output Directory: {output_dir} in gpu rank: {torch.cuda.current_device()}")
    Path(os.path.join(output_dir, "checkpoints")).mkdir(parents=True, exist_ok=True)
    
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    model = ModelWrapper(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch:02d}_"+method_name,
            auto_insert_metric_name=False,
            monitor=cfg.model.val_monitor,
            mode="min",
            save_top_k=cfg.save_top_model
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    wandb_logger = WandbLogger(save_dir=output_dir,
                               entity="kth-rpl",
                               project=f"{cfg.wandb_project_name}", 
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"),
                               log_model=(True if cfg.wandb_mode == "online" else False))
    
    trainer = pl.Trainer(logger=wandb_logger,
                         log_every_n_steps=50,
                         accelerator="gpu",
                         devices=cfg.gpus,
                         check_val_every_n_epoch=cfg.val_every,
                         gradient_clip_val=cfg.gradient_clip_val,
                         strategy="ddp_find_unused_parameters_false" if cfg.gpus > 1 else "auto",
                         callbacks=callbacks,
                         max_epochs=cfg.epochs,
                         sync_batchnorm=cfg.sync_bn)
    
    wandb_logger.watch(model, log_graph=False)
    if trainer.global_rank == 0:
        print("\n"+"-"*40)
        print("Initiating wandb and trainer successfully.  ^V^ ")
        print(f"We will use {cfg.gpus} GPUs to train the model. Check the checkpoints in {output_dir} checkpoints folder.")
        print("Total Train Dataset Size: ", len(train_dataset))
        if cfg.add_seloss is not None and cfg.loss_fn == 'seflowLoss':
            print(f"Note: We are in **self-supervised** training now. No ground truth label is used.")
            print(f"We will use these loss items in {cfg.loss_fn}: {cfg.add_seloss}")
        print("-"*40+"\n")

    # NOTE(Qingwen): search & check: def training_step(self, batch, batch_idx)
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader, ckpt_path = cfg.checkpoint)
    wandb.finish()

if __name__ == "__main__":
    main()