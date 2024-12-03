import numpy as np
import pandas as pd
import torch
from utils.RK4 import RK4_step
from utils.param_alpha import general_alpha
from models.physical_model import get_physical_model
from data_processing.context import get_context_wo_check
from utils.read_data import get_initial_position, get_two_initial_positions
from data_driven.models.hybrid_model import HybridDriftModule, Hybrid_Model
from data_driven.models.trajectory_model import TrajectoryModule
from data_driven.models.hybrid_model_w_history import HybridDriftModule_w_History
from tqdm import tqdm

def compute_position_hybrid(pos_1, time1, pretrained_model, config):

    physical_model = get_physical_model()
    alpha = general_alpha()

    #print(pos_1)
    pos_1 = pos_1.squeeze()
    xphys = physical_model(pos_1,time1,config).to('cuda')

    context = get_context_wo_check(config['PATH_WATER'],config['PATH_WIND'],config['PATH_WAVES'],pos_1[0].item(),pos_1[1].item(),time1,config['d_context'], config['npoints'])
    #context= context[:-2,:,:]
    context = torch.from_numpy(context.astype(np.float32)).to('cuda')

    xphys = torch.unsqueeze(xphys,0)

    x_pred = pretrained_model(xphys,context).to('cuda')
    return xphys, x_pred

def compute_trajectory_hybrid(config,nhours, NOAA=False):

    longitudes = np.zeros(nhours+1)
    latitudes = np.zeros(nhours+1)

    pos_1, time = get_initial_position(config['PATH_DRIFT'],NOAA)

    pos = torch.tensor([pos_1[1], pos_1[0]], dtype = torch.float) # lat, lon
    #convert pos1 to torch
    #pos = torch.from_numpy(pos_1.astype(np.float32))

    latitudes[0] = pos[0].item()
    longitudes[0] = pos[1].item()

    # Model
    module = HybridDriftModule.load_from_checkpoint(config['checkpoint_test'])
    model = module.model
    model.eval()


    for i in tqdm(range(nhours)):
        pos_phys, pos = compute_position_hybrid(pos,time,model,config)

        if pos.isnan().any():
            longitudes[i+1:] = longitudes[i]
            latitudes[i+1:] = latitudes[i]

        longitudes[i+1] = pos[0,1].item()
        latitudes[i+1] = pos[0,0].item()
        time = time + 1
        #pos = pos_phys
    

    return longitudes, latitudes

def compute_position_hybrid_w_history(pos_0, pos_1, time1, prev_context, pretrained_model, config):

    physical_model = get_physical_model()
    alpha = general_alpha()

    #print(pos_1)
    pos_1 = pos_1.squeeze()
    xphys = physical_model(pos_1,time1,config).to('cuda')

    context = get_context_wo_check(config['PATH_WATER'],config['PATH_WIND'],config['PATH_WAVES'],pos_1[0].item(),pos_1[1].item(),time1,config['d_context'], config['npoints'])
    #context= context[:-2,:,:]
    context = torch.from_numpy(context.astype(np.float32)).to('cuda')

    final_context = torch.cat((context, prev_context), 0)

    xphys = torch.unsqueeze(xphys,0)
    pos_0 = torch.unsqueeze(pos_0,0)

    x_pred = pretrained_model(xphys,pos_0,final_context).to('cuda')
    return xphys, x_pred, pos_1, context

def compute_trajectory_hybrid_w_history(config,nhours, NOAA=False):

    longitudes = np.zeros(nhours+1)
    latitudes = np.zeros(nhours+1)

    pos_0, pos_1, time = get_two_initial_positions(config['PATH_DRIFT'],NOAA)

    pos = torch.tensor([pos_1[1], pos_1[0]], dtype = torch.float) # lat, lon
    prevpos = torch.tensor([pos_0[1], pos_0[0]], dtype = torch.float) # lat, lon
    #convert pos1 to torch
    #pos = torch.from_numpy(pos_1.astype(np.float32))

    latitudes[0] = prevpos[0].item()
    longitudes[0] = prevpos[1].item()

    latitudes[1] = pos[0].item()
    longitudes[1] = pos[1].item()

    prevcontext = get_context_wo_check(config['PATH_WATER'],config['PATH_WIND'],config['PATH_WAVES'], prevpos[0].item(),prevpos[1].item(),time-1,config['d_context'], config['npoints'])
    #prevcontext= prevcontext[:-2,:,:]
    prevcontext = torch.from_numpy(prevcontext.astype(np.float32)).to('cuda')

    # Model
    module = HybridDriftModule_w_History.load_from_checkpoint(config['checkpoint_test'])
    model = module.model
    model.eval()


    for i in tqdm(range(nhours)):
        if i ==0:
            continue

        pos_phys, pos, prevpos, prevcontext = compute_position_hybrid_w_history(prevpos, pos,time,prevcontext, model,config)

        if pos.isnan().any():
            longitudes[i+1:] = longitudes[i]
            latitudes[i+1:] = latitudes[i]

        longitudes[i+1] = pos[0,1].item()
        latitudes[i+1] = pos[0,0].item()
        time = time + 1
        #pos = pos_phys
    

    return longitudes, latitudes