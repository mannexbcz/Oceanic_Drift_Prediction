import os
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from data_driven.models.hybrid_model import HybridDriftModule
from data_processing.dataset import DriftPairDataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from argparse import ArgumentParser
import yaml

if __name__ == "__main__": 
    '''parser = ArgumentParser()
    parser.add_argument("--config", type = str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)'''
    
    # Configuration
    csvfile = '../data/nextpoint_ds/next_point_dataset.csv' #TODO add config file!
    chkpt_folder = '../checkpoints'

    # Datasets
    train_dataset = DriftPairDataset(csvfile,d_context=1,npoints=32) #TODO d_context and npoints in the config
    #test_dataset = DriftPairDataset() # TODO 
    #val_dataset = DriftPairDataset() # TODO 

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=10) #TODO batchize in config
    #test_loader = DataLoader(test_dataset)
    val_loader = DataLoader(train_dataset, batch_size=1, num_workers=10)

    # Model
    model = HybridDriftModule(channels1=32, channels2=16) #TODO add channels in the config

    # Trainer
    trainer = pl.Trainer(default_root_dir=chkpt_folder, max_epochs=100,check_val_every_n_epoch=10, callbacks=[EarlyStopping(monitor="val_loss", patience = 3, mode="min")])
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    # If we want to resume training:  #TODO add to config
    # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

    # Test model
    #trainer.test(model, dataloaders=test_dataloader)

