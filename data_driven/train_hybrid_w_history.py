import os
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from data_driven.models.hybrid_model_w_history import HybridDriftModule_w_History
from data_processing.dataset import DriftPairDataset_W_Previous
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from argparse import ArgumentParser
import yaml

if __name__ == "__main__": 

    parser = ArgumentParser()
    parser.add_argument("--config", type = str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration
    csvfile_train = config['csvfile_train']
    #csvfile_test = config['csvfile_test']
    csvfile_val = config['csvfile_val']
    
    chkpt_folder = config['chkpt_folder']

    # Datasets
    #train_dataset = DriftPairDataset(csvfile,d_context=1,npoints=32)
    train_dataset = DriftPairDataset_W_Previous(csvfile_train)
    val_dataset = DriftPairDataset_W_Previous(csvfile_val)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['bs'], shuffle=True, num_workers=config['num_workers'], pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=config['bs'], num_workers=config['num_workers'])

    # Model
    model = HybridDriftModule_w_History(channels1=config['ch1'], channels2=config['ch2'], hidden1=config['hidden1'],hidden2=config['hidden2'],hidden3=config['hidden3'],hidden4=config['hidden4'],lr =config['lr']) #TODO add channels in the config

    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer(default_root_dir=chkpt_folder,min_epochs=config['min_epochs'], max_epochs=config['max_epochs'],check_val_every_n_epoch=config['check_val_every_n_epochs'], callbacks=[EarlyStopping(monitor="val_loss", patience = config['patience'], mode="min"), lr_monitor, checkpoint_callback],accelerator="gpu", devices=1) #profiler="simple",

    # Train model
    trainer.fit(model, train_loader, val_loader)

    print(trainer.callback_metrics)

    # If we want to resume training:  #TODO add to config
    # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
    
    # Test model
    #trainer.test(model, dataloaders=test_dataloader)

