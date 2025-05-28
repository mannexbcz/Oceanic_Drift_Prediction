from utils.read_data import water_interpolated, wind_interpolated, wave_interpolated, bathymetry_interpolated
from utils.convert_lats import new_latitude, new_longitude
import numpy as np
from utils.point_in_domain import check_point_boundaries
import torch
from scipy.interpolate import RegularGridInterpolator


def get_water_context(path_water, lat_init, lon_init, time_init, d = 1, npoints = 32):

    water_u_interpolation,water_v_interpolation = water_interpolated(path_water)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_water_u = np.zeros([npoints, npoints])
    context_water_v = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_water_u[i,j] = water_u_interpolation([time_init,0,lats[j],lons[i]])
            context_water_v[i,j] = water_v_interpolation([time_init,0,lats[j],lons[i]])

    return context_water_u, context_water_v

def get_wind_context(path_wind, lat_init, lon_init, time_init, d = 1, npoints = 32):

    wind_u_interpolation,wind_v_interpolation = wind_interpolated(path_wind)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_wind_u = np.zeros([npoints, npoints])
    context_wind_v = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_wind_u[i,j] = wind_u_interpolation([time_init,lats[j],lons[i]])
            context_wind_v[i,j] = wind_v_interpolation([time_init,lats[j],lons[i]])

    return context_wind_u, context_wind_v


def get_waves_context(path_waves, lat_init, lon_init, time_init, d = 1, npoints = 32):

    ust_interpolation,vst_interpolation = wave_interpolated(path_waves)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_waves_u = np.zeros([npoints, npoints])
    context_waves_v = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_waves_u[i,j] = ust_interpolation([time_init,lats[j],lons[i]])
            context_waves_v[i,j] = vst_interpolation([time_init,lats[j],lons[i]])

    return context_waves_u, context_waves_v

def get_bathymetry_context(path_baythy, lat_init, lon_init, time_init, d = 1, npoints = 32):

    elevation_interpolation = bathymetry_interpolated(path_baythy)
    d_m = d*1000 #convert in m 

    # get latitudes & longitudes coordinates of square of size d (in km) centered around init position
    lat_max = new_latitude(lat_init, lon_init, d_m/2)
    lat_min = new_latitude(lat_init, lon_init, -d_m/2)
    lon_max = new_longitude(lat_init, lon_init, d_m/2)
    lon_min = new_longitude(lat_init, lon_init, -d_m/2)

    # get grid 
    lats = np.linspace(lat_min, lat_max, npoints)
    lons = np.linspace(lon_min, lon_max, npoints)
    
    # get tensor of interpolated values
    context_bathymetry = np.zeros([npoints, npoints])

    for i in range(npoints):
        for j in range(npoints):
            context_bathymetry[i,j] = elevation_interpolation([lats[j],lons[i]])

    context_coasts = np.where(context_bathymetry>0, 0,1)

    return context_bathymetry, context_coasts


def get_context(path_water, path_wind, path_waves, init_lat, init_lon, init_time, config, d_context=1, npoints=32):

    init_lat, init_lon = check_point_boundaries(init_lat, init_lon, config['min_lat'].detach().cpu().numpy(), config['max_lat'].detach().cpu().numpy(), config['min_lon'].detach().cpu().numpy(), config['max_lon'].detach().cpu().numpy(),d_context+10)

    context_water_u, context_water_v = get_water_context(path_water, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    context_wind_u, context_wind_v = get_wind_context(path_wind, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    context_waves_u, context_waves_v = get_waves_context(path_waves, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    #context_bathymetry, context_coasts = get_bathymetry_context(path_bathy, init_lat, init_lon, init_time, d = d_context, npoints = npoints)

    context = np.stack((context_water_u,context_water_v,context_wind_u,context_wind_v,context_waves_u,context_waves_v))
    assert np.shape(context) == (6,npoints,npoints), f"Wrong shape for the context: {np.shape(context)}"

    return context


def get_context_wo_check(path_water, path_wind, path_waves, init_lat, init_lon, init_time, d_context=1, npoints=32):

    #init_lat, init_lon = check_point_boundaries(init_lat, init_lon, config['min_lat'].detach().cpu().numpy(), config['max_lat'].detach().cpu().numpy(), config['min_lon'].detach().cpu().numpy(), config['max_lon'].detach().cpu().numpy(),d_context+10)

    context_water_u, context_water_v = get_water_context(path_water, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    context_wind_u, context_wind_v = get_wind_context(path_wind, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    context_waves_u, context_waves_v = get_waves_context(path_waves, init_lat, init_lon, init_time, d = d_context, npoints = npoints)
    #context_bathymetry, context_coasts = get_bathymetry_context(path_bathy, init_lat, init_lon, init_time, d = d_context, npoints = npoints)

    context = np.stack((context_water_u,context_water_v,context_wind_u,context_wind_v,context_waves_u,context_waves_v))
    assert np.shape(context) == (6,npoints,npoints), f"Wrong shape for the context: {np.shape(context)}"

    return context


#########################################################################################################################################################################################

def extract_context_from_bigcontext(position, init_time, big_context,time_init_bigcontext, config, d=50, npoints=32):

    big_context = big_context[0,:,:,:,:]

    # Create interpolator
    lat_min = config['min_lat'].cpu().numpy()
    lat_max = config['max_lat'].cpu().numpy()
    lon_min = config['min_lon'].cpu().numpy()
    lon_max = config['max_lon'].cpu().numpy()
    points_per_degree = 10

    npoint_lats = int((lat_max-lat_min)*points_per_degree)
    npoint_lons = int((lon_max-lon_min)*points_per_degree)

    # get grid 
    lats_full = np.squeeze(np.linspace(lat_min, lat_max, npoint_lats))
    lons_full = np.squeeze(np.linspace(lon_min, lon_max, npoint_lons))
    time = np.arange(72) + time_init_bigcontext.cpu().numpy()
    variables = np.arange(6)  # 6 variables

    interpolator = RegularGridInterpolator((variables, time, lons_full, lats_full), big_context)

    lat_init, lon_init = position[0], position[1]
    d_m = d*1000 #convert to m
    lat_max = new_latitude(lat_init, lon_init, d_m / 2)
    lat_min = new_latitude(lat_init, lon_init, -d_m / 2)
    lon_max = new_longitude(lat_init, lon_init, d_m / 2)
    lon_min = new_longitude(lat_init, lon_init, -d_m / 2)

    # get grid 
    lats = np.squeeze(np.linspace(lat_min, lat_max, npoints))
    lons = np.squeeze(np.linspace(lon_min, lon_max, npoints))

    # Create meshgrid for latitudes and longitudes
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Flatten the spatial grid
    points_spatial = np.array([lon_grid.ravel(), lat_grid.ravel()]).T
    n_points = points_spatial.shape[0]

    time_points = np.full(n_points, init_time)
    var_points = np.repeat(variables, n_points)
    lat_lon_repeated = np.tile(points_spatial, (len(variables), 1))

    # Combine time, spatial, and variable dimensions into a single points array
    points = np.column_stack((var_points, np.repeat(time_points, len(variables)), lat_lon_repeated))

    # Clip the latitude and longitude values to grid bounds
    points[:, 2] = np.clip(points[:, 2], lons_full[0], lons_full[-1])  # Latitude
    points[:, 3] = np.clip(points[:, 3], lats_full[0], lats_full[-1])  # Longitude

    # Interpolate all variables
    interpolated_values = interpolator(points)

    # Reshape the results: (npoints, npoints, variables)
    context = interpolated_values.reshape(len(variables), npoints, npoints)    
    context = torch.from_numpy(context.astype(np.float32))

    return context

def extract_context_from_bigcontext_torch(positions, init_times, big_context, time_init_bigcontext, config, d=50, npoints=32, device='cuda'):
    """
    Extracts contexts for a batch of positions and initial times using PyTorch for GPU execution.
    
    Args:
        positions (torch.Tensor): Tensor of shape (batch_size, 2) containing [latitude, longitude] pairs.
        init_times (torch.Tensor): Tensor of shape (batch_size,) containing initial times.
        big_context (torch.Tensor): The full context tensor of shape (variables, time, longitude, latitude).
        time_init_bigcontext (int): Start time of the big context.
        config (dict): Configuration dictionary containing min/max latitude and longitude.
        d (float): Size of the context box in kilometers. Defaults to 50.
        npoints (int): Number of points in the context box grid. Defaults to 32.
        device (str): Device to run computations on. Defaults to 'cuda'.
    
    Returns:
        torch.Tensor: Batch of extracted contexts of shape (batch_size, npoints, npoints, variables).
    """
    # Configuration for the grid
    lat_min_b, lat_max_b = config['min_lat'], config['max_lat']
    lon_min_b, lon_max_b = config['min_lon'], config['max_lon']

    variables = torch.arange(6, device=device)
    
    # Flatten the big context to prepare for interpolation
    #big_context = big_context.to(device)  # Ensure big_context is on GPU
    #big_context_flat = big_context.permute(0, 1, 3, 2).contiguous()  # Shape: (variables, time, latitude, longitude)
    
    # Prepare outputs
    batch_size = positions.size(0)
    contexts = torch.empty((batch_size, len(variables), npoints, npoints), device=device)

    # Loop over the batch
    for b in range(batch_size):
        big_contextb = big_context[b]
        lat_init, lon_init = positions[b]
        init_time = init_times[b]
        lat_min = lat_min_b[b]
        lat_max = lat_max_b[b]
        lon_min = lon_min_b[b]
        lon_max = lon_max_b[b]
        time_init_bigcontextb = time_init_bigcontext[b]

        lat_init = torch.clamp(lat_init, min=lat_min, max=lat_max)  # Clip latitude
        lon_init = torch.clamp(lon_init, min=lon_min, max=lon_max)  # Clip longitude
        init_time = torch.clamp(init_time, min=time_init_bigcontextb, max=time_init_bigcontextb+71) # Clip time

        d_m = d * 1000  # Convert d to meters
        lat_max = new_latitude(lat_init.item(), lon_init.item(), d_m / 2)
        lat_min = new_latitude(lat_init.item(), lon_init.item(), -d_m / 2)
        lon_max = new_longitude(lat_init.item(), lon_init.item(), d_m / 2)
        lon_min = new_longitude(lat_init.item(), lon_init.item(), -d_m / 2)
        
        # Create grid for the context box
        lats = torch.linspace(lat_min, lat_max, npoints, device=device)
        lons = torch.linspace(lon_min, lon_max, npoints, device=device)
        lon_grid, lat_grid = torch.meshgrid(lons, lats, indexing='ij')

        # Prepare interpolation points
        time_idx = int((init_time - time_init_bigcontextb).item())
        #time_idx = torch.tensor([time_idx], device=device).clamp(0, 72)

        # Normalize the query points to [0, 1] scale for grid_sample
        norm_lon = ((lon_grid - lon_min) / (lon_max - lon_min) * 2 - 1).unsqueeze(0)
        norm_lat = ((lat_grid - lat_min) / (lat_max - lat_min) * 2 - 1).unsqueeze(0)

        # Combine the normalized points into a single grid
        grid = torch.stack((norm_lon, norm_lat), dim=-1)  # Shape: (1, npoints, npoints, 2)

        # Interpolate all variables
        for var in range(len(variables)):
            variable_slice = big_contextb[var, time_idx].unsqueeze(0).unsqueeze(0).float()
            context = torch.nn.functional.grid_sample(
                variable_slice, grid, mode='bilinear', align_corners=True
            )
            contexts[b, var] = context.squeeze()

    # Permute to desired output shape
    #contexts = contexts.permute(0, 2, 3, 1)  # (batch_size, npoints, npoints, variables)
    
    return contexts


def extract_context_from_bigcontext_torch_no_batch(positions, init_time, big_context, time_init_bigcontextb, config, d=50, npoints=32, device='cuda'):
    # Configuration for the grid
    lat_min, lat_max = config['min_lat'], config['max_lat']
    lon_min, lon_max = config['min_lon'], config['max_lon']

    variables = torch.arange(6, device=device)
    
    # Flatten the big context to prepare for interpolation
    #big_context = big_context.to(device)  # Ensure big_context is on GPU
    #big_context_flat = big_context.permute(0, 1, 3, 2).contiguous()  # Shape: (variables, time, latitude, longitude)
    
    contexts = torch.empty((len(variables), npoints, npoints), device=device)

    big_contextb = big_context
    lat_init, lon_init = positions

    lat_init = torch.clamp(lat_init, min=lat_min, max=lat_max)  # Clip latitude
    lon_init = torch.clamp(lon_init, min=lon_min, max=lon_max)  # Clip longitude
    init_time = torch.clamp(init_time, min=time_init_bigcontextb, max=time_init_bigcontextb+71) # Clip time

    d_m = d * 1000  # Convert d to meters
    lat_max = new_latitude(lat_init.item(), lon_init.item(), d_m / 2)
    lat_min = new_latitude(lat_init.item(), lon_init.item(), -d_m / 2)
    lon_max = new_longitude(lat_init.item(), lon_init.item(), d_m / 2)
    lon_min = new_longitude(lat_init.item(), lon_init.item(), -d_m / 2)
        
    # Create grid for the context box
    lats = torch.linspace(lat_min, lat_max, npoints, device=device)
    lons = torch.linspace(lon_min, lon_max, npoints, device=device)
    lon_grid, lat_grid = torch.meshgrid(lons, lats, indexing='ij')

    # Prepare interpolation points
    time_idx = int((init_time - time_init_bigcontextb).item())
    #time_idx = torch.tensor([time_idx], device=device).clamp(0, 72)

    # Normalize the query points to [0, 1] scale for grid_sample
    norm_lon = ((lon_grid - lon_min) / (lon_max - lon_min) * 2 - 1).unsqueeze(0)
    norm_lat = ((lat_grid - lat_min) / (lat_max - lat_min) * 2 - 1).unsqueeze(0)

    # Combine the normalized points into a single grid
    grid = torch.stack((norm_lon, norm_lat), dim=-1)  # Shape: (1, npoints, npoints, 2)

        # Interpolate all variables
    for var in range(len(variables)):
        variable_slice = big_contextb[var, time_idx].unsqueeze(0).unsqueeze(0).float()
        context = torch.nn.functional.grid_sample(
            variable_slice, grid, mode='bilinear', align_corners=True
        )
        contexts[var] = context.squeeze()

    # Permute to desired output shape
    #contexts = contexts.permute(0, 2, 3, 1)  # (batch_size, npoints, npoints, variables)
    
    return contexts
