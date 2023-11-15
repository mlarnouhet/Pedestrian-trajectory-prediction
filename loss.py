import torch
import numpy as np

def get_coefficients(outputs):
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr

def Gaussian2DLikelihood(outputs, targets, pred_length):
    seq_length = outputs.shape[0]
    obs_length = seq_length - pred_length
    mux, muy, sx, sy, corr = get_coefficients(outputs)
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2
    result = torch.exp(-z/(2*negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    result = result / denom
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    loss = 0
    counter = 0

    id_list = range(outputs.shape[1])
    for frame in range(obs_length, seq_length):
        for id in id_list:
            loss = loss + result[frame, id]
            counter = counter + 1
    if counter != 0:
        return loss / counter
    else:
        return loss