"""
# Created: 2023-08-09 10:28
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Output the evaluation results, go for local evaluation or online evaluation
"""

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from scripts.network.dataloader import HDF5Dataset
from scripts.pl_model import ModelWrapper

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
    
    checkpoint_params = DictConfig(torch.load(cfg.checkpoint)["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output + f"-{cfg.av2_mode}-v{cfg.leaderboard_version}"
    cfg.model.update(checkpoint_params.cfg.model)
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
    print(f"\n---LOG[eval]: Loaded model from {cfg.checkpoint}. The model is {checkpoint_params.cfg.model.name}.\n")

    wandb_logger = WandbLogger(save_dir=output_dir,
                               entity="kth-rpl",
                               project=f"deflow-eval", 
                               name=f"{cfg.output}",
                               offline=(cfg.wandb_mode == "offline"))
    
    trainer = pl.Trainer(logger=wandb_logger, devices=1)
    # NOTE(Qingwen): search & check: def eval_only_step_(self, batch, res_dict)
    trainer.validate(model = mymodel, \
                     dataloaders = DataLoader(HDF5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}", eval=True, leaderboard_version=cfg.leaderboard_version), batch_size=1, shuffle=False))
    wandb.finish()

if __name__ == "__main__":
    main()