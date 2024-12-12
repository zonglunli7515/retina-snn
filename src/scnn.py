import h5py

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

def minmax_scaler(train, test):
    train_min = train.min()
    train_max = train.max()
    return (train - train_min) / (train_max - train_min), (test - train_min) / (train_max - train_min)

class VectorizeData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.size(0)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y
    
train_filepath = "../data/train_mov3.hdf5"
test_filepath = "../data/test_mov3.hdf5"

train_data = h5py.File(train_filepath, 'r')
X_train = train_data['mov3_trn_x']
y_train = train_data['mov3_trn_y']
r_train = train_data['mov3_trn_r']

test_data = h5py.File(test_filepath, 'r')
X_test = test_data['mov3_tst_x']
y_test = test_data['mov3_tst_y']
r_test = test_data['mov3_tst_r']

thr = 3
X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train = np.array(y_train)
r_train = np.array(r_train)
X_train = X_train.permute(2, 0, 1)
X_train = X_train.unsqueeze(1)
r_train = r_train.T
#r_train = np.where(r_train>thr ,1, 0)
r_train = 33/30 * r_train
r_train = torch.tensor(r_train, dtype=torch.float)

X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_test = np.array(y_test)
r_test = np.array(r_test)
X_test = X_test.permute(2, 0, 1)
X_test = X_test.unsqueeze(1)
r_test = r_test.T
#r_test = np.where(r_test>thr, 1, 0)
r_test = 33/30 * r_test
r_test = torch.tensor(r_test, dtype=torch.float)


X_train, X_test = minmax_scaler(X_train, X_test)
bs = 20

ds_train = VectorizeData(X_train, r_train)
train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=False)
ds_test = VectorizeData(X_test, r_test)
test_loader = DataLoader(ds_test, batch_size=bs, shuffle=True, drop_last=False)


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

beta = 0.9
num_steps = 33
spike_grad = surrogate.fast_sigmoid(slope=25)
#  Initialize Network
net = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=1, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*7*7, 80),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),

                    ).to(device)

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

"""
for step in range(num_steps):
    spk_out, mem_out = net(data)
"""

def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)

spk_rec, mem_rec = forward_pass(net, num_steps, data)
spk_rec = spk_rec.sum(dim=0)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.9, 0.999))
loss_val = torch.zeros((1), dtype=torch.float, device=device)
loss_val = loss_fn(spk_rec, targets)

optimizer.zero_grad()
loss_val.backward()
optimizer.step()