from helpers import *
import torch
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt

def plot_trajectory(observations):
    plt.figure()
    plt.scatter(observations[:, 0].cpu().detach().numpy(), observations[:, 1].cpu().detach().numpy(), color='b', label="Observations")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title("Trajectory Visualization in camera frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")


def plot_scene(scene):
    plt.figure
    for frame in range(scene.shape[0]):
        plt.scatter(scene[frame, :, 0].cpu().detach().numpy(), scene[frame, :, 1].cpu().detach().numpy(), color='b')
    plt.legend()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title("Scene Visualization in camera frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")






def evaluate_social(model, val_dataloader, batch_size_val, nb_in, hidden_size, device):
    model.eval()
    mad = 0
    fad = 0
    for idx, (scenes, peds, grids) in enumerate(tqdm(val_dataloader)):
        for (scene, ped, grid) in zip(scenes, peds, grids):
            scene = torch.tensor(scene).to(device)
            hidden_states = Variable(torch.zeros(scene.shape[1], hidden_size)).to(device)
            cell_states = Variable(torch.zeros(scene.shape[1], hidden_size)).to(device)
            predicted_trajectories = model.infer(scene[:nb_in], grid[:nb_in], hidden_states, cell_states, [700, 500])
            mad_iter, fad_iter = distance_metrics(scene[nb_in:,:,1:3], predicted_trajectories)
            mad += mad_iter
            fad += fad_iter

    print(f"average mad: {mad/ (len(val_dataloader) * batch_size_val)}")
    print(f"average fad: {fad / (len(val_dataloader) * batch_size_val)}")
    model.train()

    return mad, fad




def evaluate_regular(model, val_dataloader, nb_in, hidden_size, device):
    model.eval()
    mad = 0
    fad = 0
    for idx, src in enumerate(tqdm(val_dataloader)):
        src = src.float().to(device)
        hidden_states = torch.zeros(src.shape[0], hidden_size).to(device)
        cell_states = torch.zeros(src.shape[0], hidden_size).to(device)
        predicted_trajectories = model.infer(src[:, :nb_in, :], hidden_states, cell_states)
        mad_iter, fad_iter, _ = distance_metrics_regular(src[:, nb_in:, :], predicted_trajectories)
        mad += mad_iter
        fad += fad_iter

    print(f"average mad: {mad/ len(val_dataloader)}")
    print(f"average fad: {fad / len(val_dataloader)}")
    model.train()

    return mad, fad