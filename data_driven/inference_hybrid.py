import os
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from data_driven.models.hybrid_model import HybridDriftModule
from data_processing.dataset import DriftPairDataset

if __name__ == "__main__": 
    # Model
    model = HybridDriftModule(channels1=32, channels2=16).load_from_checkpoint("path_to_chkpt") #TODO add channels in the config¨
    model.eval()
    x_pred = model(x)

    # To get the hyperparameters: 
    model.nameofparameter

    
    