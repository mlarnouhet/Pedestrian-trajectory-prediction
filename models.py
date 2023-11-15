import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import itertools
from torch.distributions.multivariate_normal import MultivariateNormal

def plot_trajectory(observations):
    plt.figure()
    plt.scatter(observations[:, 0].cpu().detach().numpy(), observations[:, 1].cpu().detach().numpy(), color='b', label="Observations")
    plt.legend()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.title("Trajectory Visualization in camera frame")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

class SocialLSTM(nn.Module):
    def __init__(self, device, input_size, output_size, embedding_size, neighborhood_size, grid_size, hidden_size, sequence_length, nb_in):
        super(SocialLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.device = device
        self.nb_in = nb_in
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.grid_size = grid_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.output_size = output_size
        self.neighborhood_size = neighborhood_size
        self.cell = nn.LSTMCell(2*self.embedding_size, self.hidden_size)
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.hidden_size, self.embedding_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

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

        return grid

    def getSocialTensor(self, grid, hidden_states):
        nb_peds = grid.shape[0]
        social_tensor = Variable(torch.zeros(nb_peds, self.grid_size*self.grid_size, self.hidden_size)).to(self.device)

        for ped in range(nb_peds):
            social_tensor[ped] = torch.mm(torch.t(grid[ped]), hidden_states)
        social_tensor = social_tensor.view(nb_peds, self.grid_size*self.grid_size*self.hidden_size)

        return social_tensor

    def sample(self, params):
        mean, sx, sy, corr = params[:, 0:2], torch.exp(params[:, 2]), torch.exp(params[:, 3]), torch.tanh(params[:, 4])
        nb_peds = params.shape[0]
        sample = torch.zeros(nb_peds,2)

        for ped in range(nb_peds):
            cov = torch.tensor([[sx[ped]**2, sx[ped]*sy[ped]*corr[ped]], [sx[ped]*sy[ped]*corr[ped], sy[ped]**2]]).to(self.device)
            distrib = MultivariateNormal(loc=mean[ped,:], covariance_matrix=cov)
            sample[ped,:] = distrib.rsample()

        return sample

    def infer(self, scene, grids, hidden_states, cell_states, dimensions):
        nb_peds = scene.shape[1]
        output_coords = torch.zeros(self.sequence_length-self.nb_in+1, nb_peds, 2).to(self.device)

        for frame in range(self.nb_in):
            frame_state = scene[frame]
            grid = torch.tensor(grids[frame]).to(self.device)
            social_tensor = self.getSocialTensor(grid, hidden_states)
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(frame_state[:, 1:3].to(torch.float32))))
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            h_state, c_state = self.cell(concat_embedded, (hidden_states, cell_states))
            hidden_states = h_state
            cell_states = c_state

        output = self.sample(self.output_layer(h_state))
        output_coords[0,:,:] = output
        frame_state = torch.concatenate((frame_state[:,0].unsqueeze(1),output.to(self.device)),1)

        for frame in range(1,self.sequence_length-self.nb_in+1):
            grid = self.compute_grids(frame_state, self.neighborhood_size, self.grid_size, dimensions).to(self.device)
            social_tensor = self.getSocialTensor(grid, hidden_states)
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(frame_state[:, 1:3].to(torch.float32))))
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            h_state, c_state = self.cell(concat_embedded, (hidden_states, cell_states))
            hidden_states = h_state
            cell_states = c_state
            output = self.sample(self.output_layer(h_state))
            output_coords[frame, :, :] = output
            frame_state = torch.concatenate((frame_state[:,0].unsqueeze(1),output.to(self.device)),1)

        return output_coords

    def forward(self, scene, grids, hidden_states, cell_states, dimensions):
        nb_peds = scene.shape[1]
        outputs = torch.zeros(self.sequence_length, nb_peds, self.output_size).to(self.device)

        for frame in range(self.sequence_length):
            frame_state = scene[frame]
            grid = torch.tensor(grids[frame]).to(self.device)
            social_tensor = self.getSocialTensor(grid, hidden_states)
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(frame_state[:,1:3].to(torch.float32))))
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            h_state, c_state = self.cell(concat_embedded, (hidden_states, cell_states))
            outputs[frame] = self.output_layer(h_state)
            hidden_states = h_state
            cell_states = c_state

        return outputs, hidden_states, cell_states



class RegularLSTM(nn.Module):
    def __init__(self, device, input_size, output_size, embedding_size, hidden_size, sequence_length, nb_in):
        super(RegularLSTM, self).__init__()
        self.sequence_length = sequence_length
        self.device = device
        self.nb_in = nb_in
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.output_size = output_size
        self.cell = nn.LSTMCell(self.embedding_size, self.hidden_size)
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

    def infer(self, trajectory, hidden_states, cell_states):
        output_coords = torch.zeros(trajectory.shape[0], self.sequence_length - self.nb_in + 1, 2).to(self.device)

        for frame in range(self.nb_in):
            position = trajectory[:, frame, :]
            input_embedded = self.relu(self.input_embedding_layer(position.to(torch.float32)))
            h_state, c_state = self.cell(input_embedded, (hidden_states, cell_states))
            hidden_states = h_state
            cell_states = c_state

        output = self.output_layer(h_state)
        output_coords[:, 0, :] = output[:,0:2]
        position = output

        for frame in range(1, self.sequence_length - self.nb_in + 1):
            input_embedded = self.relu(self.input_embedding_layer(position.to(torch.float32)))
            h_state, c_state = self.cell(input_embedded, (hidden_states, cell_states))
            hidden_states = h_state
            cell_states = c_state
            output = self.output_layer(h_state)
            output_coords[:, frame, :] = output[:,0:2]
            position = output

        return output_coords

    def forward(self, trajectory, hidden_states, cell_states):
        outputs = torch.zeros(trajectory.shape[0], self.sequence_length, self.output_size).to(self.device)
        for frame in range(self.sequence_length):
            position = trajectory[:, frame, :]
            input_embedded = self.relu(self.input_embedding_layer(position.to(torch.float32)))
            h_state, c_state = self.cell(input_embedded, (hidden_states, cell_states))
            outputs[:, frame, :] = self.output_layer(h_state)
            hidden_states = h_state
            cell_states = c_state

        return outputs, hidden_states, cell_states

