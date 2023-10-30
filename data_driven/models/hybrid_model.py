from typing import Any
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.physical_model import get_physical_model

class Hybrid_Model(nn.Module):
    def __init__(self, channels1 = 32, channels2 = 16) -> None:
        super().__init__()
        self.physical_model = get_physical_model()

        def block(in_channels, out_channels):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1, padding_mode='reflect'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            return layers
        
        self.data_driven_model = nn.Sequential(
            *block(8,channels1),
            *block(channels1,channels2),
            *block(channels2,1), 
            nn.Flatten(0,-1),
            nn.Linear(16,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.Tanh()
        )

        self.final_part = nn.Sequential(
            nn.Linear(34,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Tanh()
        )

    def forward(self, init_position, init_time, context, dict_path):
        xphys = self.physical_model(init_position, init_time, dict_path)
        xNN_part = self.data_driven_model(context)
        x = torch.cat((xphys,xNN_part))
        xNN = self.final_part(x)
        return torch.add(xphys, xNN) 


class HybridDriftModule(pl.LightningModule):
    def __init__(self,channels1=32, channels2=16):
        super().__init__()
        self.save_hyperparameters()
        self.model = Hybrid_Model(channels1, channels2)

    def training_step(self, batch, batch_idx):
        init_position, final_position, init_time, context, dict_path = batch
        xpred = self.model(init_position, init_time, context, dict_path)
        loss = None #TODO:define Loss
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        init_position, final_position, init_time, context, dict_path = batch
        xpred = self.model(init_position, init_time, context, dict_path)
        loss = None #TODO:define Loss
        self.log("test_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        init_position, final_position, init_time, context, dict_path = batch
        xpred = self.model(init_position, init_time, context, dict_path)
        loss = None #TODO:define Loss
        self.log("val_loss", loss)
        return 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer







