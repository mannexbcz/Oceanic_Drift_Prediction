from typing import Any
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.physical_model import get_physical_model
from haversine import haversine
from data_driven.losses import haversine_loss, cumulative_lagrangian_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

def extract_nth_point(data_dict, n):
    result_dict = {}
    for key, values in data_dict.items():
        if isinstance(values, list):
            result_dict[key] = values[n]
        elif isinstance(values, dict):
            result_dict[key] = {k: v[n] for k, v in values.items()}
        elif isinstance(values, torch.Tensor):
            result_dict[key] = values[n].item()
    return result_dict

class Trajectory_Model(nn.Module):
    def __init__(self, channels1 = 32, channels2 = 16, hidden1=128,hidden2=64,hidden3=128,hidden4=64) -> None:
        super().__init__()
        self.physical_model = get_physical_model()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def block(in_channels, out_channels):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1, padding_mode='reflect'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            return layers
        
        self.data_driven_model = nn.Sequential(
            #*block(6,16),
            *block(6,channels1),
            *block(channels1,channels2),
            *block(channels2,1), 
            nn.Flatten(1,-1),
            nn.Linear(16,hidden1), #16
            nn.ReLU(),
            nn.Linear(hidden1,hidden2),
            nn.ReLU(),
            nn.Linear(hidden2,32),
            nn.Tanh()
        )

        self.final_part = nn.Sequential(
            nn.Linear(34,hidden3),
            nn.ReLU(),
            nn.Linear(hidden3,hidden4),
            nn.ReLU(),
            nn.Linear(hidden4,2),
            nn.Tanh()
        )

    def forward(self, pos0, time0, config, context):
        #print(time0)
        #print(config)
        #print(context)
        xphys = torch.zeros_like(pos0)
        for i in range(pos0.size(dim=0)):
            #xphys[i,:] = torch.unsqueeze(self.physical_model(torch.squeeze(pos0[i,:]), time0[i], extract_nth_point(config, i)).detach().to(self.device),dim=0) #(need to do .detach()?)
            xphys[i,:] = torch.unsqueeze(self.physical_model(torch.squeeze(pos0[i,:]), torch.Tensor([time0]), config).detach().to(self.device),dim=0) #(need to do .detach()?)
        xphys = xphys.to(self.device)
        xNN_part = self.data_driven_model(context)
        #xphys = torch.squeeze(xphys)
        x = torch.cat((xphys,xNN_part), dim=-1)
        xNN = self.final_part(x)
        return xNN+xphys #torch.add(xphys, xNN) 


class TrajectoryModule(pl.LightningModule):
    def __init__(self,channels1=32, channels2=16,hidden1=128,hidden2=64,hidden3=128,hidden4=64, lr = 1e-3,bs=32):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = Trajectory_Model(channels1, channels2, hidden1,hidden2,hidden3,hidden4)
        #self.loss = nn.MSELoss()
        self.writer = SummaryWriter()
        self.bs = bs

    def training_step(self, batch, batch_idx):

        pos0, pos1, pos2, pos3, time0, config, context = batch

        xpred1 = self.model(pos0,time0,config,context)
        xpred2 = self.model(xpred1,time0+1,config,context)
        xpred3 = self.model(xpred2,time0+1,config,context)
        
        loss = cumulative_lagrangian_loss(pos0,pos1,pos2,pos3,xpred1,xpred2,xpred3)

        self.log("train_loss", loss,prog_bar=True,on_epoch=True, on_step=True, batch_size = self.bs)
        return loss
    
    def test_step(self, batch, batch_idx):
        #init_position, final_position, init_time, context, dict_path = batch
        initial_position,xphys, final_position, context = batch
        #xpred = self.model(init_position, init_time, context, dict_path)
        xpred = self.model(xphys,context)
        loss = None
        self.log("test_loss", loss)
        return torch.Tensor([loss])
    
    def validation_step(self, batch, batch_idx):
        pos0, pos1, pos2, pos3, time0, config, context = batch

        xpred1 = self.model(pos0,time0,config,context)
        xpred2 = self.model(xpred1,time0+1,config,context)
        xpred3 = self.model(xpred2,time0+1,config,context)


        loss = cumulative_lagrangian_loss(pos0,pos1,pos2,pos3,xpred1,xpred2,xpred3)

        self.log("val_loss", loss,prog_bar=True,on_epoch=True, on_step=True,batch_size = self.bs)
        self.log("hp_metric", loss)
        return 
    
    def on_training_epoch_end(self, training_step_outputs):
        # Log gradients and weights at the end of each epoch
        for name, param in self.parameters():
            self.writer.add_histogram(name, param, global_step=self.current_epoch)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,momentum = 0.9)
        #scheduler = ReduceLROnPlateau(optimizer, patience=10)
        return optimizer
        '''return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }'''







