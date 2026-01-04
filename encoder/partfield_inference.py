from partfield.config import default_argument_parser, setup
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning
import torch
import glob
import os, sys
import numpy as np
import random

def predict(cfg):
    seed_everything(cfg.seed)

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    checkpoint_callbacks = [ModelCheckpoint(
        monitor="train/current_epoch",
        dirpath=cfg.output_dir,
        filename="{epoch:02d}",
        save_top_k=100,
        save_last=True,
        every_n_epochs=cfg.save_every_epoch,
        mode="max",
        verbose=True
    )]

    trainer = Trainer(devices=-1,
                      accelerator="gpu",
                      precision="16-mixed",
                      strategy=DDPStrategy(find_unused_parameters=True),
                      max_epochs=cfg.training_epochs,
                      log_every_n_steps=1,
                      limit_train_batches=3500,
                      limit_val_batches=None,
                      callbacks=checkpoint_callbacks
                     )

    from partfield.model_trainer_pvcnn_only_demo import Model
    model = Model(cfg)        

    if cfg.remesh_demo:
        cfg.n_point_per_face = 10

    checkpoint = torch.load(cfg.continue_ckpt, map_location='cpu')
    state_dict = checkpoint['state_dict']
    if all(k.startswith('model.') for k in state_dict.keys()):
        new_state_dict = {k[len('model.'):]: v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict, strict=True)

    trainer.predict(model)
        
def main():
    parser = default_argument_parser()
    args = parser.parse_args()
    cfg = setup(args, freeze=False)
    predict(cfg)
    
if __name__ == '__main__':
    main()