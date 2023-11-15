from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from torch.autograd import Variable
from models import SocialLSTM
from helpers import *
from loss import *
from datasets import *
from evaluate import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 64
batch_size_train = 16
batch_size_val = 16
hidden_size = 128
learning_rate = 1e-4
nb_epochs = 100
nb_in = 8
nb_out_train = 12
nb_out_val = 12
input_size = 2
output_size = 5
grid_size = 4
neighborhood_size = 32
sequence_length = nb_in + nb_out_train - 1

train_dataset = SocialTrajNetDataset("eth", nb_in, nb_out_train, "train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size_train, shuffle=True, collate_fn = custom_collate)
val_dataset = SocialTrajNetDataset("eth", nb_in, nb_out_val, "test")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size_val, shuffle=True, collate_fn = custom_collate)

model = SocialLSTM(device, input_size, output_size, embedding_size, neighborhood_size, grid_size, hidden_size, sequence_length, nb_in).to(device)
optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=0.005)

for n in range(nb_epochs):
   print(n)
   mad_list = []
   fad_list = []
   loss_tot = 0
   for idx, (scenes, peds, grids) in enumerate(train_dataloader):
       loss_batch = 0
       for (scene, ped, grid) in zip(scenes, peds, grids):
           scene = torch.tensor(scene).to(device)
           hidden_states = Variable(torch.zeros(scene.shape[1], hidden_size)).to(device)
           cell_states = Variable(torch.zeros(scene.shape[1], hidden_size)).to(device)
           outputs, _, _ = model(scene[:-1], grid[:-1], hidden_states, cell_states, [700, 500])
           loss = Gaussian2DLikelihood(outputs.to(device), scene[1:,:,1:3], nb_out_train)
           optimizer.zero_grad()
           loss.backward()
           nn.utils.clip_grad_norm_(model.parameters(), 10)
           optimizer.step()
           loss_batch += loss.item()

       loss_tot += loss_batch

   print(loss_tot/(len(train_dataloader)*batch_size_train))
   if n == 10:
       mad, fad = evaluate_social(model, val_dataloader, batch_size_val, nb_in, hidden_size, device)
       mad_list.append(mad)
       fad_list.append(fad)


print("end")

