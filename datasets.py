from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from helpers import *

class SocialTrajNetDataset(Dataset):

    def __init__(self, data_dir, nb_in, nb_out, mode):
        self.data_dir = data_dir
        self.nb_in = nb_in
        self.nb_out = nb_out
        self.mode = mode
        self.scenes, self.active_peds = self.get_scenes()
        self.shape_list = [scene.shape for scene in self.scenes]
        self.grids = self.get_grids(self.scenes)

    def compute_grids(self, frame_state, neighborhood_size, grid_size, dimensions):
        nb_peds = frame_state.shape[0]
        width, height = dimensions[0], dimensions[1]
        grid = torch.zeros((nb_peds, nb_peds, grid_size ** 2))
        width_bound, height_bound = (neighborhood_size / (width * 1.0)) * 2, (neighborhood_size / (height * 1.0)) * 2

        for ped_id in range(nb_peds):
            current_x, current_y = frame_state[ped_id, 1], frame_state[ped_id, 2]
            width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
            height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2

            for other_ped_id in range(nb_peds):
                if frame_state[other_ped_id, 0] == frame_state[ped_id, 0]:
                    continue
                other_x, other_y = frame_state[other_ped_id, 1], frame_state[other_ped_id, 2]
                if other_x >= width_high or other_x < width_low or other_y >= height_high or other_y < height_low:
                    continue
                cell_x = int(torch.floor(((other_x - width_low) / width_bound) * grid_size))
                cell_y = int(torch.floor(((other_y - height_low) / height_bound) * grid_size))
                if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                    continue
                grid[ped_id, other_ped_id, cell_x + cell_y * grid_size] = 1

        return grid.cpu().detach().numpy()

    def get_grids(self, scenes):
        grids = []
        for scene in scenes:
            scene_grids = []
            for frame in range(scene.shape[0]):
                scene_grids.append(self.compute_grids(torch.tensor(scene[frame]), 32, 4, [700, 500]))
            grids.append(scene_grids)
        return grids

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        scene = self.scenes[index]
        grids = self.grids[index]
        peds_list = self.active_peds[index]
        return scene, peds_list, grids

    def get_scenes(self):
        scene_list = []
        active_peds_list = []
        dataset_path = "datasets" + "\\" + self.data_dir + "\\" + self.mode

        for file in os.listdir(dataset_path):
            df = pd.read_csv(os.path.join(dataset_path,file), header=None, sep="\t").values

            for starting_frame in np.unique(df[:,0])[:-(self.nb_in+self.nb_out-1)]:
                scene = df[(df[:, 0] < starting_frame + 10 * (self.nb_in + self.nb_out)) & (df[:, 0] >= starting_frame)]
                scene = augment_scene(scene)
                subscene_list = []
                peds_per_frame_list = []
                pedestrian_ids = np.unique(scene[:,1])
                for frame in np.unique(scene[:,0]):
                    subscene = np.zeros((len(pedestrian_ids),3))
                    subscene[:,0] = pedestrian_ids
                    frame_state = scene[scene[:,0] == frame]
                    peds_per_frame_list.append([np.where(pedestrian_ids == i)[0][0] for i in np.unique(frame_state[:,1])])
                    for n, id in enumerate(pedestrian_ids):
                        if id in np.unique(frame_state[:,1]):
                            subscene[n,1:3] = frame_state[frame_state[:,1] == id][0][2:4]
                        else:
                            subscene[n,1:3] = [0,0]
                    subscene_list.append(subscene)
                next_scene = np.stack(subscene_list,0)
                if next_scene.shape[0] == self.nb_in+self.nb_out:
                    filtered_scene, filtered_peds_per_frame = filter_scene(next_scene)
                    if filtered_scene.shape[1] > 1:
                        filtered_scene[:,:,1:3] -= filtered_scene[0,:,1:3]
                        scene_list.append(filtered_scene)
                        active_peds_list.append(filtered_peds_per_frame)
        return scene_list, active_peds_list


class RegularTrajNetDataset(Dataset):

    def __init__(self, data_dir, nb_in, nb_out, mode):
        self.data_dir = data_dir
        self.nb_in = nb_in
        self.nb_out = nb_out
        self.mode = mode
        self.trajectories = self.get_trajectories()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        if (self.mode == "train"):
            trajectory = torch.tensor(augment(self.trajectories[index]["pos"]))
        else:
            trajectory = torch.tensor(self.trajectories[index]["pos"])
        return trajectory

    def get_trajectories(self):
        thr = 3
        trajectories = []
        dataset_path = "datasets" + "\\" + self.data_dir + "\\" + self.mode

        for file in os.listdir(dataset_path):
            df = pd.read_csv(os.path.join(dataset_path,file), header=None, sep="\t")
            nb_ped = int(max(df.iloc[:,1]))

            for ped_id in range(1,nb_ped+1):
                ped_traj = df[df.iloc[:,1] == ped_id]
                if (ped_traj.shape[0] > self.nb_in + self.nb_out):
                    for i in range(ped_traj.shape[0] - (self.nb_in + self.nb_out)+1):
                        trajectories.append({"start": min(ped_traj[0][i:(self.nb_in + self.nb_out)+i]), "end": max(ped_traj[0][i:(self.nb_in + self.nb_out)+i]), "id":ped_id, "pos":ped_traj.iloc[i:(self.nb_in + self.nb_out)+i,[2,3]].to_numpy()})

        return trajectories














