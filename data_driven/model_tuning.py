import os
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from data_driven.models.hybrid_model import HybridDriftModule
from data_driven.models.hybrid_model_w_history import HybridDriftModule_w_History
from data_processing.dataset import DriftPairDataset, DriftPairDataset_Wo_Computation,DriftPairDataset_W_Previous
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from argparse import ArgumentParser
from lightning.pytorch import seed_everything
import yaml
import optuna

csvfile_train= '../data/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_train_with_previous.csv'
csvfile_val= '../data/NOAA/nextpoint_ds/contexts/pt32d50/next_point_dataset_val_with_previous.csv'
chkpt_folder = '../checkpoints/NOAA/gridsearch_w_history'
bs = 32
num_workers= 32
min_epochs = 150
max_epochs = 1000
check_val_every_n_epochs =  1
patience = 20
seed = 42

def objective(trial: optuna.trial.Trial):

    seed_everything(seed, workers = True)
    # Parameters
    channel1 = trial.suggest_int("channel1", 16,64)
    channel2 = trial.suggest_int("channel2", 16,64)
    hidden1 = trial.suggest_int("hidden1", 64,256)
    hidden2 = trial.suggest_int("hidden2", 64,256)
    hidden3 = trial.suggest_int("hidden3", 64,256)
    hidden4 = trial.suggest_int("hidden4", 64,256)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    # Datasets
    train_dataset = DriftPairDataset_Wo_Computation(csvfile_train)
    val_dataset = DriftPairDataset_Wo_Computation(csvfile_val)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=num_workers)

    # Model
    model = HybridDriftModule(channels1=channel1, channels2=channel2, hidden1=hidden1,hidden2=hidden2,hidden3=hidden3,hidden4=hidden4,lr =lr) #TODO add channels in the config

    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch')
    trainer = pl.Trainer(default_root_dir=chkpt_folder,min_epochs=min_epochs, max_epochs=max_epochs,check_val_every_n_epoch=check_val_every_n_epochs,enable_progress_bar=False,enable_model_summary = False, callbacks=[EarlyStopping(monitor="val_loss", patience = patience, mode="min"), lr_monitor, checkpoint_callback],accelerator="gpu", devices=1) #profiler="simple",


    trainer.fit(model, train_loader, val_loader)


    return trainer.callback_metrics["val_loss_epoch"].item()


def objective_w_history(trial: optuna.trial.Trial):

    seed_everything(seed, workers = True)
    # Parameters
    channel1 = trial.suggest_int("channel1", 16,64)
    channel2 = trial.suggest_int("channel2", 16,64)
    hidden1 = trial.suggest_int("hidden1", 64,256)
    hidden2 = trial.suggest_int("hidden2", 64,256)
    hidden3 = trial.suggest_int("hidden3", 64,256)
    hidden4 = trial.suggest_int("hidden4", 64,256)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    # Datasets
    train_dataset = DriftPairDataset_W_Previous(csvfile_train)
    val_dataset = DriftPairDataset_W_Previous(csvfile_val)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True) 
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=num_workers)

    # Model
    model = HybridDriftModule_w_History(channels1=channel1, channels2=channel2, hidden1=hidden1,hidden2=hidden2,hidden3=hidden3,hidden4=hidden4,lr =lr) #TODO add channels in the config

    # Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch')
    trainer = pl.Trainer(default_root_dir=chkpt_folder,min_epochs=min_epochs, max_epochs=max_epochs,check_val_every_n_epoch=check_val_every_n_epochs,enable_progress_bar=False,enable_model_summary = False, callbacks=[EarlyStopping(monitor="val_loss", patience = patience, mode="min"), lr_monitor, checkpoint_callback],accelerator="gpu", devices=1) #profiler="simple",


    trainer.fit(model, train_loader, val_loader)


    return trainer.callback_metrics["val_loss_epoch"].item()

if __name__ == "__main__": 

    sampler = optuna.samplers.TPESampler(seed=17)

    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective_w_history, n_trials=100)

    print('Best params: ', study.best_params)