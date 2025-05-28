import os
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
import lightning.pytorch as pl
from data_driven.models.trajectory_model import TrajectoryModule, TrajectoryModule_Dyn
from data_processing.dataset import TrajectoryDataset,TrajectoryDataset_Dynamic
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
    train_dataset = TrajectoryDataset(csvfile_train)
    val_dataset = TrajectoryDataset(csvfile_val)
    '''train_dataset = TrajectoryDataset_Dynamic(csvfile_train)
    val_dataset = TrajectoryDataset_Dynamic(csvfile_val)'''
    

    # Dataloaders
    train_sampler = SequentialSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)

    train_batch_sampler = BatchSampler(train_sampler, batch_size=config['bs'], drop_last=False)
    val_batch_sampler = BatchSampler(val_sampler, batch_size=config['bs'], drop_last=False)

    '''train_loader = DataLoader(train_dataset, batch_size=config['bs'], shuffle=False, num_workers=config['num_workers'], pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=config['bs'], num_workers=config['num_workers'])'''

    train_loader = DataLoader(train_dataset, batch_sampler = train_batch_sampler, num_workers=config['num_workers'], pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_sampler = val_batch_sampler, num_workers=config['num_workers'])

    # Model
    model = TrajectoryModule(channels1=config['ch1'], channels2=config['ch2'], hidden1=config['hidden1'],hidden2=config['hidden2'],hidden3=config['hidden3'],hidden4=config['hidden4'],lr =config['lr'])

    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',save_last=True)
    trainer = pl.Trainer(default_root_dir=chkpt_folder,min_epochs=config['min_epochs'], max_epochs=config['max_epochs'],check_val_every_n_epoch=config['check_val_every_n_epochs'], callbacks=[EarlyStopping(monitor="val_loss", patience = config['patience'], mode="min"), lr_monitor, checkpoint_callback],accelerator="gpu", devices=1) #profiler="simple",

    # Train model
    trainer.fit(model, train_loader, val_loader, ckpt_path='~/checkpoints/MasterThesis/lightning_logs/version_13/checkpoints/epoch=77-step=84162.ckpt')

    print(trainer.callback_metrics)


