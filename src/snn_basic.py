import h5py
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
r_train = np.where(r_train>thr ,1, 0)
r_train = torch.tensor(r_train, dtype=torch.float)

X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
y_test = np.array(y_test)
r_test = np.array(r_test)
X_test = X_test.permute(2, 0, 1)
X_test = X_test.unsqueeze(1)
r_test = r_test.T
r_test = np.where(r_test>thr, 1, 0)
r_test = torch.tensor(r_test, dtype=torch.float)

X_train, X_test = minmax_scaler(X_train, X_test)
bs = 20

ds_train = VectorizeData(X_train, r_train)
train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=False)
ds_test = VectorizeData(X_test, r_test)
test_loader = DataLoader(ds_test, batch_size=bs, shuffle=True, drop_last=False)

# Network Architecture
num_inputs = X_train.size(-1)**2
num_hidden = 1000

# Temporal Dynamics
num_steps = 10
beta = 0.95
num_outputs = r_train.size(-1)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define Network
class BasicSNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(bs, -1))
    spk_sum = output.sum(dim=0)
    spk_sum[spk_sum > 0] = 1
    acc = (spk_sum == targets).sum().item() / spk_sum.numel()

    sum_both = spk_sum + targets
    sum_both = torch.where(sum_both > 1, 1, 0)
    sensitivity = sum_both.sum() / targets.sum()

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        print(f"Train set sensitivity for a single minibatch: {sensitivity*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
        print(f"Test set sensitivity for a single minibatch: {sensitivity*100:.2f}%")

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")
        
# Load the network onto CUDA if available
net = BasicSNN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.9, 0.999))
loss = nn.BCEWithLogitsLoss()

"""
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)
spk_rec, mem_rec = net(data.view(bs, -1))

loss_val = torch.zeros((1), dtype=torch.float, device=device)

for step in range(num_steps):
    loss_val += loss(mem_rec[step], targets)
print(f"Training loss: {loss_val.item():.3f}")
print_batch_accuracy(data, targets, train=True)

# clear previously stored gradients
optimizer.zero_grad()

# calculate the gradients
loss_val.backward()

# weight update
optimizer.step()

# calculate new network outputs using the same data
spk_rec, mem_rec = net(data.view(bs, -1))

# initialize the total loss value
loss_val = torch.zeros((1), dtype=torch.float, device=device)

# sum loss at every step
for step in range(num_steps):
  loss_val += loss(mem_rec[step], targets)

print(f"Training loss: {loss_val.item():.3f}")
print_batch_accuracy(data, targets, train=True)
"""

num_epochs = 5
loss_hist = []
test_loss_hist = []
counter = 0

last_step = False

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(bs, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=torch.float, device=device)
        if last_step:
            loss_val = loss(mem_rec[-1], targets)
        else:
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(bs, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=torch.float, device=device)
            if last_step:
                test_loss = loss(test_mem[-1], test_targets)
            else:
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 5 == 0:
                train_printer(
                    data, targets, epoch,
                    counter, iter_counter,
                    loss_hist, test_loss_hist,
                    test_data, test_targets)
            counter += 1
            iter_counter +=1