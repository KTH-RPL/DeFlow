"""
# Created: 2023-08-09 10:28
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Description: Output the evaluation results, go for local evaluation or online evaluation
"""

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from scripts.network.dataloader_flow4D import HDF5Dataset
from scripts.pl_model_flow4D import ModelWrapper
from datetime import datetime

@hydra.main(version_base=None, config_path="conf", config_name="eval_flow4D")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    current_time = datetime.now().strftime("%m%d_%H%M%S")

    if cfg.av2_mode == 'test':
        cfg.save_res = True
        print('test mode')
    elif cfg.av2_mode == 'val':
        print('val mode')
    else:
        raise ValueError(f"Invalid value for cfg.av2_mode: {cfg.av2_mode}")


    parts = cfg.checkpoint.split('/')
    wandb_index = parts.index('wandb')
    relevant_parts = parts[wandb_index + 1:-1] 

    file_parts = parts[-1].split('.')[0].split('_')
    save_folder = f"{relevant_parts[0]}_{file_parts[0]}_{file_parts[1]}"
    save_folder = save_folder + '_' + current_time
    output_dir = os.path.join('/data/jiehao/MambaFlow_bucket/results', save_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
    
    checkpoint_params = DictConfig(torch.load(cfg.checkpoint)["hyper_parameters"])

    cfg.output = save_folder
    cfg.model.update(checkpoint_params.cfg.model)
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)

    trainer = pl.Trainer(devices=1)
    trainer.validate(model = mymodel, \
                     dataloaders = DataLoader(HDF5Dataset(cfg.dataset_path + f"/{cfg.av2_mode}", n_frames=cfg.num_frames, eval=True), batch_size=1, shuffle=False))
    wandb.finish()

if __name__ == "__main__":
    main()