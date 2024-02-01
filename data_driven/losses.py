import torch
import math

def haversine_loss(pred, target, epsSq = 1.e-13, epsAs = 1.e-7):   # add optional epsilons to avoid singularities

    lat1, lon1 = torch.split(pred, 1, dim=1)
    lat2, lon2 = torch.split(target, 1, dim=1)
    r = 6371  # Radius of Earth in kilometers
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    # return tensor.mean(2 * r * torch.asin(torch.sqrt(a)))
    # "+ (1.0 - a**2) * epsSq" to keep sqrt() away from zero
    # "(1.0 - epsAs) *" to keep asin() away from plus-or-minus one
    return torch.Tensor.mean(2 * r * torch.asin ((1.0 - epsAs) * torch.sqrt (a + (1.0 - a**2) * epsSq)))


def LDA_loss_step(pred,target,init_pos, k1=0.5,k2=0.5):

    '''norm_pred = torch.linalg.vector_norm(pred-init_pos)
    norm_targ = torch.linalg.vector_norm(target-init_pos)'''

    norm_pred = haversine_loss(pred,init_pos)
    norm_targ = haversine_loss(target,init_pos)

    dotprod = torch.dot(torch.flatten(pred-init_pos),torch.flatten(target-init_pos))

    #loss = k1 * (math.sqrt((norm_targ-norm_pred)**2)/(norm_pred+norm_targ)) + 0.5*k2*(1-dotprod/(norm_pred*norm_targ))
    loss = math.sqrt((norm_targ-norm_pred)**2)/(norm_pred+norm_targ)

    return loss


def haversine_loss_cosine(pred, target,init_pos, epsSq = 1.e-13, epsAs = 1.e-7):
    norm_pred = haversine_loss(pred,init_pos)
    norm_targ = haversine_loss(target,init_pos)

    dotprod = torch.dot(torch.flatten(pred-init_pos),torch.flatten(target-init_pos))

    lat1, lon1 = torch.split(pred, 1, dim=1)
    lat2, lon2 = torch.split(target, 1, dim=1)
    r = 6371  # Radius of Earth in kilometers
    phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
    delta_phi, delta_lambda = torch.deg2rad(lat2-lat1), torch.deg2rad(lon2-lon1)
    a = torch.sin(delta_phi/2)**2 + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda/2)**2
    # return tensor.mean(2 * r * torch.asin(torch.sqrt(a)))
    # "+ (1.0 - a**2) * epsSq" to keep sqrt() away from zero
    # "(1.0 - epsAs) *" to keep asin() away from plus-or-minus one
    return torch.Tensor.mean(2 * r * torch.asin ((1.0 - epsAs) * torch.sqrt (a + (1.0 - a**2) * epsSq))) + 0.5*(1-dotprod/(norm_pred*norm_targ))


def cumulative_lagrangian_loss(pos0,pos1,pos2,pos3,pred1,pred2,pred3):
    ds1 = haversine_loss(pred1,pos1)
    ds2 = haversine_loss(pred2,pos2)
    ds3 = haversine_loss(pred3,pos3)

    dl1 = haversine_loss(pos0,pos1)
    dl2 = haversine_loss(pos0,pos2)
    dl3 = haversine_loss(pos0,pos3)

    L1 = dl1
    L2 = dl1 + dl2
    L3 = dl1 + dl2 + dl3

    return (ds1 + ds2 + ds3)/(L1 + L2 + L3)