from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from torch.autograd import Variable
from models import *
from helpers import *
from loss import *
from datasets import *
from evaluate import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size = 64
batch_size_train = 64
batch_size_val = 64
hidden_size = 128
learning_rate = 1e-4
nb_epochs = 100
nb_in = 8
nb_out_train = 12
nb_out_val = 12
input_size = 2
output_size = 2
sequence_length = nb_in + nb_out_train - 1

train_dataset = RegularTrajNetDataset("eth", nb_in, nb_out_train, "train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size_train, shuffle=True)
val_dataset = RegularTrajNetDataset("eth", nb_in, nb_out_val, "test")
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size_val, shuffle=True)

model = RegularLSTM(device, input_size, output_size, embedding_size, hidden_size, sequence_length, nb_in).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
loss_fn = nn.MSELoss()
mad_list = []
fad_list = []

for n in range(nb_epochs):
    print(n)
    if n == 20:
        optimizer.param_groups[0]['lr'] /= 10
    if n == 50:
        optimizer.param_groups[0]['lr'] /= 10


    train_loss = 0
    for idx, src in enumerate(tqdm(train_dataloader)):
        src = src.float().to(device)
        hidden_states = torch.zeros(src.shape[0], hidden_size).to(device)
        cell_states = torch.zeros(src.shape[0], hidden_size).to(device)
        output, _, _ = model(src[:,:-1,:], hidden_states, cell_states)
        loss = loss_fn(output, src[:, 1:, :])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        train_loss += loss

    print(f"avg_train_loss = {train_loss / len(train_dataloader)}")
    if n % 10 == 0:
        mad, fad = evaluate_regular(model, val_dataloader, nb_in, hidden_size, device)
    mad_list.append(mad)
    fad_list.append(fad)



print("end")

