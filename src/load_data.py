import os, random
import h5py
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import reservoirpy
from reservoirpy.nodes import Reservoir, Ridge
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def rolling_window(arr, history, time_axis=-1):

    if time_axis == 0:
        arr = arr.T
    elif time_axis == -1:
        pass
    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')
    
    assert history >= 1, "'window' must be least 1."
    assert history < arr.shape[-1], "'window' is longer than array"

    #with strides 
    shape = arr.shape[:-1] + (arr.shape[-1] - history, history)
    strides = arr.strides + (arr.strides[-1], )
    arr = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr, 1, 0)
    else:
        return arr

# Define the CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=0)  # 5x5 filter
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # Flatten
        x = x.view(-1, 3 * 43 * 43)        
        return x


if __name__ == "__main__":

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

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = np.array(y_train)
    r_train = np.array(r_train)
    X_train = X_train.permute(2, 0, 1)
    X_train = X_train.unsqueeze(1)
    r_train = r_train.T
    r_train[r_train>0] = 1

    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = np.array(y_test)
    r_test = np.array(r_test)
    X_test = X_test.permute(2, 0, 1)
    X_test = X_test.unsqueeze(1)
    r_test = r_test.T
    r_test[r_test>0] = 1

    model = SimpleCNN()
    #a = model(X[0:1])
    input_train = model(X_train)
    input_train = input_train.detach().numpy()
    
    n_hidden = 500
    # Create a reservoir
    reservoir = Reservoir(units=n_hidden, lr=0.9, connectivity=0.1, input_scaling=1, spectral_radius=0.9)
    # Run the reservoir with input data
    train_states = reservoir.run(input_train)
    reservoir = reservoir.reset()

    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(train_states, r_train)

    input_test = model(X_test)
    input_test = input_test.detach().numpy()
    test_states = reservoir.run(input_test)
    reservoir = reservoir.reset()
    
    preds = clf.predict_proba(test_states)
    """
    # Create a Ridge regression readout (output layer)
    readout = Ridge(ridge=1e-6)

    # Assemble the ESN
    esn = reservoir >> readout

    # Train the ESN with multidimensional target data
    esn.fit(input_train, r_train)  

    # testing
    input_test = model(X_test)
    input_test = input_test.detach().numpy()
    preds = esn.run(input_test)
    """