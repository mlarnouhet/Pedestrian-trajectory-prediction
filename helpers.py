import os
import random
import numpy as np
import pandas as pd
import scipy.spatial

def augment(trajectory):
    p = 1/3
    proba = random.uniform(0, 1)
    if (proba <= p):
        if (np.random.rand() < 0.5):
            return np.concatenate((-np.expand_dims(trajectory[:,0],1),np.expand_dims(trajectory[:,1],1)), 1)
        else:
            return np.concatenate((np.expand_dims(trajectory[:,0],1),-np.expand_dims(trajectory[:,1],1)), 1)
    elif (proba > p and proba <= 2*p):
        angle = np.random.randint(90)*np.pi/180
        rotation = np.matrix([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        return np.transpose(rotation @ np.transpose(trajectory))

    else:
        return trajectory

def augment_scene(scene):
    p = 1/3
    proba = random.uniform(0, 1)
    if (proba <= p):
        if (np.random.rand() < 0.5):
            return np.concatenate((scene[:,0:2], -np.expand_dims(scene[:, 2], 1), np.expand_dims(scene[:, 3], 1)), 1)
        else:
            return np.concatenate((scene[:, 0:2], np.expand_dims(scene[:, 2], 1), -np.expand_dims(scene[:, 3], 1)), 1)
    elif (proba > p and proba <= 2 * p):
        angle = np.random.randint(90) * np.pi / 180
        rotation = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.array(np.concatenate((scene[:,0:2], np.transpose(rotation @ np.transpose(scene[:,2:4]))), 1))
    else:
        return scene

def filter_scene(scene):
    filtered_ids = []
    thr = 4
    for id in range(scene.shape[1]):
        if ((scene[0,id,1] != 0 or scene[0,id,2] != 0) and (scene[-1,id,1] != 0 or scene[-1,id,2] != 0) and ((np.abs(scene[0,id,1]-scene[-1,id,1]) > thr) or (np.abs(scene[0,id,2]-scene[-1,id,2]) > thr))):
            filtered_ids.append(id)
    filtered_scene = scene[:,filtered_ids,:]
    return filtered_scene, [range(filtered_scene.shape[1]) for _ in range(scene.shape[0])]





def compute_datasets_boundaries(dataset_dir):
    boundaries_list = {}
    for file in os.listdir(dataset_dir):
        df = pd.read_csv(os.path.join(dataset_dir, file), header=None, sep="\t").values
        dataset_name = file.split(".")[0]
        boundaries_list[dataset_name] = [np.max(np.absolute(df[:,2])), np.max(np.absolute(df[:,3]))]
    return boundaries_list


def custom_collate(data):
    scene, peds_list, grids = zip(*data)
    return list(scene), list(peds_list), list(grids)


def distance_metrics(target, preds):
    target = target.cpu().detach().numpy()
    preds = preds.cpu().detach().numpy()
    mad_list = []
    fad_list = []

    for ped_id in range(preds.shape[1]):
        displacement_tot = 0
        for frame in range(preds.shape[0]):
            displacement = np.sqrt(np.sum((target[frame, ped_id,:] - preds[frame, ped_id, :])**2))
            displacement_tot += displacement
        mad_list.append(displacement_tot/preds.shape[0])
        fad_list.append(displacement)

    mad = np.mean(mad_list)
    fad = np.mean(fad_list)
    return mad, fad



def distance_metrics_regular(target, preds):

    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(target[i, j].cpu().detach(), preds[i, j].cpu().detach())

    mad = errors.mean()
    fad = errors[:, -1].mean()

    return mad, fad, errors