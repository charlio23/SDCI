import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import TensorDataset


class RealGRN(Dataset):

    def __init__(self, root_dir, test=False):
        self.root_dir = root_dir
        mode = '_test' if test else '_train'
        data_string = 'grn_dataset' + mode + '.npy'
        self.dataset = np.load(os.path.join(root_dir, data_string))[...,None]

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, i):
        return self.dataset[i]


class GRN(Dataset):

    def __init__(self, root_dir, num_atoms, num_dims=1, hidden_states=False, masked=False):
        self.root_dir = root_dir
        self.num_atoms = num_atoms
        self.num_dims = num_dims
        self.masked = masked
        off_diag_idx = get_off_diag_idx(num_atoms)
        data_string = 'gene_regulation_dataset_1000.npy'
        state_string = 'gene_regulation_states_1000.npy'
        graph_string = 'gene_regulation_gt_1000.npy'
        if self.masked:
            data_string = 'gene_regulation_dataset_full.npy'
            state_string = 'gene_regulation_states_full.npy'
            mask_string = 'gene_regulation_mask_full.npy'
            self.mask = np.load(os.path.join(root_dir, mask_string))
        self.dataset = np.load(os.path.join(root_dir, data_string))[...,None]
        self.graph = np.load(os.path.join(root_dir, graph_string))
        self.state = np.load(os.path.join(root_dir, state_string))
        edges = np.reshape(self.graph, [num_atoms ** 2]).astype(int)
        self.edges = edges[off_diag_idx]

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, i):
        if self.masked:
            return self.dataset[i], self.state[i], self.edges, self.mask[i]
        return self.dataset[i], self.state[i], self.edges

class NBADataset(Dataset):

    def __init__(self, root_dir, idx_path=None, hidden_states=True):
        self.root_dir = root_dir
        self.idx_path = idx_path
        self.hidden_states = hidden_states
        self.seq_path = os.path.join(root_dir, 'trajectories')
        self.state_path = os.path.join(root_dir, 'state_trajectories')
        self.file_list = sorted(os.listdir(self.seq_path))
        self.idx_list = [] if self.idx_path is None else np.load(idx_path)

    def __len__(self):
        return len(self.file_list) if len(self.idx_list)==0 else len(self.idx_list)

    def __getitem__(self, i):
        idx = i if self.idx_path is None else self.idx_list[i]
        sample = np.load(os.path.join(self.seq_path, self.file_list[idx]))
        # sample: (200, 11, 6): 11 = 10 players + 1 ball; 6 = xyz, vx, vy, vz
        sample[:,:,0] -= 50
        sample[:,:,0] /= 50
        sample[:,:,1] -= 25
        sample[:,:,1] /= 25
        sample[:,:,2] -= 4
        sample[:,:,2] /= 4
        sample[:,:,3] /= 30
        sample[:,:,4] /= 30
        sample[:,:,5] /= 5
        if not self.hidden_states:
            sample_states = np.load(os.path.join(self.state_path, self.file_list[idx]))[...,None]
            shape = list(sample_states.shape)
            shape[-1] = 4
            state_onehot = torch.zeros(shape)
            state_onehot.scatter_(-1,torch.from_numpy(sample_states).long(), torch.ones(sample_states.shape))
            sample = np.concatenate([sample, state_onehot.float().numpy()], axis=-1)
        # Capped at 100, for forecasting we may want to remove the cap.
        return sample
    

class NBADatasetOld(Dataset):

    def __init__(self, root_dir, idx_path=None):
        self.root_dir = root_dir
        self.idx_path = idx_path
        self.file_list = sorted(os.listdir(root_dir))
        self.idx_list = [] if self.idx_path is None else np.load(idx_path)

    def __len__(self):
        return len(self.file_list) if len(self.idx_list)==0 else len(self.idx_list)

    def __getitem__(self, i):
        if self.idx_path is None:
            file = np.load(os.path.join(
            self.root_dir, self.file_list[i]))
        else:
            idx = self.idx_list[i]
            file = np.load(os.path.join(
                self.root_dir, self.file_list[idx]))
        sample = file
        # sample: (200, 11, 6): 11 = 10 players + 1 ball; 6 = xyz, vx, vy, vz
        sample[:,:,0] -= 50
        sample[:,:,0] /= 50
        sample[:,:,1] -= 25
        sample[:,:,1] /= 25
        sample[:,:,2] -= 4
        sample[:,:,2] /= 4
        sample[:,:,3] /= 30
        sample[:,:,4] /= 30
        sample[:,:,5] /= 5
        return sample[:100,:,:]


def augment_trajectory_full(data, rotations='one'):
    """
    Perform full data augmentation on particle trajectories, including
    90°, 180°, and 270° rotations, and flips.

    Args:
        data (numpy.ndarray): The input trajectory data of shape (T, N, 4), where:
            T = number of time steps,
            N = number of particles,
            Each particle has [x, y, v_x, v_y].

    Returns:
        numpy.ndarray: Augmented data, combining the original and transformed trajectories.
    """
    # Define augmentation functions
    def rotate_90(trajectory):
        # 90° Counterclockwise: (x, y) -> (-y, x), (v_x, v_y) -> (-v_y, v_x)
        augmented = trajectory.copy()
        augmented[..., 0], augmented[..., 1] = -trajectory[..., 1], trajectory[..., 0]  # x, y
        augmented[..., 2], augmented[..., 3] = -trajectory[..., 3], trajectory[..., 2]  # v_x, v_y
        return augmented

    def rotate_180(trajectory):
        # 180° Rotation: negate all components
        augmented = trajectory.copy()
        augmented[..., 0:4] *= -1
        return augmented

    def rotate_270(trajectory):
        # 270° Counterclockwise: (x, y) -> (y, -x), (v_x, v_y) -> (v_y, -v_x)
        augmented = trajectory.copy()
        augmented[..., 0], augmented[..., 1] = trajectory[..., 1], -trajectory[..., 0]  # x, y
        augmented[..., 2], augmented[..., 3] = trajectory[..., 3], -trajectory[..., 2]  # v_x, v_y
        return augmented

    def flip_vertical(trajectory):
        # Flip vertically: negate y-components
        augmented = trajectory.copy()
        augmented[..., 1] *= -1  # y
        augmented[..., 3] *= -1  # v_y
        return augmented

    def flip_horizontal(trajectory):
        # Flip horizontally: negate x-components
        augmented = trajectory.copy()
        augmented[..., 0] *= -1  # x
        augmented[..., 2] *= -1  # v_x
        return augmented

    # Apply augmentations
    augmented_data = [data]  # Original data
    augmented_data.append(rotate_180(data))
    if rotations=='all':
        augmented_data.append(rotate_90(data))
        augmented_data.append(rotate_270(data))
    augmented_data.append(flip_vertical(data))
    augmented_data.append(flip_horizontal(data))

    # Stack augmented data
    return np.concatenate(augmented_data, axis=0)

def load_springs_data(root_dir, train_idx, suffix, num_atoms, num_states=2, one_hot=False, hidden_states=False, split='train', data_size=-1, norm_tuple=None, augment=False):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    
    loc_train = np.load(os.path.join(root_dir, "loc_" + split + suffix + ".npy"))
    vel_train = np.load(os.path.join(root_dir, "vel_" + split + suffix + ".npy"))
    state_train = np.load(os.path.join(root_dir, "state_" + split + suffix + ".npy"))
    edges_train = np.load(os.path.join(root_dir, "edges_" + split + suffix + ".npy"))
    if data_size != -1:
        loc_train = loc_train[:data_size]
        vel_train = vel_train[:data_size]
        state_train = state_train[:data_size]
        edges_train = edges_train[:data_size]
    if norm_tuple is None:
        loc_max = loc_train.max()
        loc_min = loc_train.min()
        vel_max = vel_train.max()
        vel_min = vel_train.min()
    else:
        loc_max = norm_tuple[0]
        loc_min = norm_tuple[1]
        vel_max = norm_tuple[2]
        vel_min = norm_tuple[3]
    # Exclude self edges
    off_diag_idx = get_off_diag_idx(num_atoms)

    feat, state, edges = data_preparation(
        loc_train,
        vel_train,
        state_train,
        edges_train,
        loc_min,
        loc_max,
        vel_min,
        vel_max,
        off_diag_idx,
        num_atoms,
        num_states,
        one_hot,
        hidden_states,
        'all' if (num_states==4 or 'collision' in root_dir) else 'one',
        augment
    )
    if train_idx is not None:
        idx_list = np.load(train_idx)
        feat, state, edges = feat[idx_list], state[idx_list], edges[idx_list]
    train_data = TensorDataset(feat, state, edges)
    return train_data, (loc_max, loc_min, vel_max, vel_min)


def get_off_diag_idx(num_atoms):
    return np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms],
    )

def data_preparation(
    loc,
    vel,
    state,
    edges,
    loc_min,
    loc_max,
    vel_min,
    vel_max,
    off_diag_idx,
    num_atoms,
    num_states,
    one_hot,
    hidden_states,
    rotations,
    augment=False
):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    # Normalize to [-1, 1]
    loc = normalize(loc, loc_min, loc_max)
    vel = normalize(vel, vel_min, vel_max)

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc = np.transpose(loc, [0, 3, 1, 2])
    vel = np.transpose(vel, [0, 3, 1, 2])
    state = np.transpose(state[:,:,None,:], [0, 3, 1, 2])
    if one_hot and not hidden_states:
        shape = list(state.shape)
        shape[-1] = edges.shape[1]
        state_onehot = torch.zeros(shape)
        state_onehot.scatter_(-1,torch.from_numpy(state).long(),torch.ones(state.shape))
        feat = np.concatenate([loc, vel, state_onehot.numpy()], axis=3)
    elif not hidden_states:
        feat = np.concatenate([loc, vel, state], axis=3)
    else:
        feat = np.concatenate([loc, vel], axis=3)
    edges = np.reshape(edges, [edges.shape[0], edges.shape[1], num_atoms ** 2]).astype(int)
    if feat.shape[-1]==4 and augment:
        feat = augment_trajectory_full(feat, rotations=rotations)
    num_augments = feat.shape[0]//state.shape[0]
    feat = torch.FloatTensor(feat)
    state = torch.LongTensor(np.tile(state, (num_augments, 1, 1, 1)))
    edges = torch.LongTensor(np.tile(edges, (num_augments, 1, 1)))
    
    edges = edges[:, :, off_diag_idx]
    
    return feat, state, edges

def normalize(x, x_min, x_max):
    return (x - x_min) * 2 / (x_max - x_min) - 1
        

if __name__ == "__main__":


    trainDS = load_springs_data('datasets/data', '_springs5', 5)
    train = torch.utils.data.DataLoader(trainDS, shuffle=True, batch_size=16)
    sample, edges = next(iter(train))
    print(sample.size())
    print(edges.size())