"""
# Created: 2023-07-12 19:30
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    train_dataset = HDF5Dataset(cfg.dataset_path + "/train")
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              collate_fn=collate_fn_pad,
                              pin_memory=True)
    val_loader = DataLoader(HDF5Dataset(cfg.dataset_path + "/val"),
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            collate_fn=collate_fn_pad,
                            pin_memory=True)
                            
    
    # count gpus, overwrite gpus
    cfg.gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    model_name = cfg.model.name
    Path(os.path.join(output_dir, "checkpoints")).mkdir(parents=True, exist_ok=True)
    
    cfg = DictConfig(OmegaConf.to_container(cfg, resolve=True))
    model = ModelWrapper(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch:02d}_"+model_name,
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
        print("-"*40+"\n")

    # NOTE(Qingwen): search & check: def training_step(self, batch, batch_idx)
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader, ckpt_path = cfg.checkpoint)
    wandb.finish()

if __name__ == "__main__":
    main()