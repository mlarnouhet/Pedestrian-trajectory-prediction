import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np

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

def plot_scene_and_pred_scene(scene, pred_scene, nb_in):
    plt.figure
    for frame in range(nb_in - 1):
        plt.scatter(scene[frame, :, 1].cpu().detach().numpy(), scene[frame, :, 2].cpu().detach().numpy(), color='b')
    plt.scatter(scene[nb_in - 1, :, 1].cpu().detach().numpy(), scene[nb_in - 1, :, 2].cpu().detach().numpy(), color='b',
                label='input positions')

    for frame in range(nb_in, scene.shape[0] - 1):
        plt.scatter(scene[frame, :, 1].cpu().detach().numpy(), scene[frame, :, 2].cpu().detach().numpy(), color='g')
    plt.scatter(scene[scene.shape[0] - 1, :, 1].cpu().detach().numpy(), scene[scene.shape[0] - 1, :, 2].cpu().detach().numpy(), color='g', label='ground_truth')

    for frame in range(scene.shape[0] - nb_in - 1):
        plt.scatter(pred_scene[frame, :, 0].cpu().detach().numpy(), pred_scene[frame, :, 1].cpu().detach().numpy(), color='r')
    plt.scatter(pred_scene[scene.shape[0] - nb_in - 1, :, 0].cpu().detach().numpy(), pred_scene[scene.shape[0] - nb_in - 1, :, 1].cpu().detach().numpy(), color='r', label='social lstm')

    plt.legend()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title("Scene Visualization in camera frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

def plot_prediction(scene, ground_truth, prediction, nb_in, nb_out):
    plt.figure
    for frame in range(nb_in):
        plt.scatter(scene[frame, :, 0].cpu().detach().numpy(), scene[frame, :, 1].cpu().detach().numpy(), color='b', label = 'observations')
    for frame in range(nb_in+1, nb_in+nb_out):
        plt.scatter(prediction[frame, :, 0].cpu().detach().numpy(), prediction[frame, :, 1].cpu().detach().numpy(), color='r', label = 'ground truth')
        plt.scatter(ground_truth[frame, :, 0].cpu().detach().numpy(), ground_truth[frame, :, 1].cpu().detach().numpy(), color='g', label = 'prediction')
    plt.legend()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title("Scene Visualization in camera frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")



def plot_all_models(dataloader, social_lstm, regular_lstm, device, hidden_size, nb_in):
    (scene, ped, grid) = next(iter(dataloader))
    scene = scene[0]
    grid = grid[0]
    scene = torch.tensor(scene).to(device)
    hidden_states = Variable(torch.zeros(scene.shape[1], hidden_size)).to(device)
    cell_states = Variable(torch.zeros(scene.shape[1], hidden_size)).to(device)
    predicted_trajectories_social = social_lstm.infer(scene[:nb_in], grid[:nb_in], hidden_states, cell_states, [700, 500])

    predicted_trajectories_regular = torch.tensor(np.zeros(predicted_trajectories_social.shape))
    for traj_id in range(scene.shape[1]):
        src = scene[:, traj_id, :]
        hidden_state = torch.zeros(1, hidden_size).to(device)
        cell_state = torch.zeros(1, hidden_size).to(device)
        predicted_trajectories_regular[:,traj_id,:] = regular_lstm.infer(src[:nb_in, 1:3].unsqueeze(0), hidden_state, cell_state).squeeze(0)

    plt.figure
    for frame in range(nb_in-1):
        plt.scatter(scene[frame, :, 1].cpu().detach().numpy(), scene[frame, :, 2].cpu().detach().numpy(), color='b')
    plt.scatter(scene[nb_in-1, :, 1].cpu().detach().numpy(), scene[nb_in-1, :, 2].cpu().detach().numpy(), color='b', label='input positions')

    for frame in range(nb_in, scene.shape[0]-1):
        plt.scatter(scene[frame, :, 1].cpu().detach().numpy(), scene[frame, :, 2].cpu().detach().numpy(), color='g')
    plt.scatter(scene[scene.shape[0]-1, :, 1].cpu().detach().numpy(), scene[scene.shape[0]-1, :, 2].cpu().detach().numpy(), color='g', label='ground_truth')

    for frame in range(scene.shape[0]-nb_in-1):
        plt.scatter(predicted_trajectories_social[frame, :, 0].cpu().detach().numpy(), predicted_trajectories_social[frame, :, 1].cpu().detach().numpy(), color='r')
    plt.scatter(predicted_trajectories_social[scene.shape[0]-nb_in-1, :, 0].cpu().detach().numpy(), predicted_trajectories_social[scene.shape[0]-nb_in-1, :, 1].cpu().detach().numpy(), color='r', label='social lstm')

    for frame in range(scene.shape[0]-nb_in-1):
        plt.scatter(predicted_trajectories_regular[frame, :, 0].cpu().detach().numpy(), predicted_trajectories_regular[frame, :, 1].cpu().detach().numpy(), color='k')
    plt.scatter(predicted_trajectories_regular[scene.shape[0]-nb_in-1, :, 0].cpu().detach().numpy(), predicted_trajectories_regular[scene.shape[0]-nb_in-1, :, 1].cpu().detach().numpy(), color='k', label='regular lstm')

    plt.legend()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title("Outputs of the different models")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")


