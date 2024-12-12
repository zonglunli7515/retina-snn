import h5py
import numpy as np

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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=1, padding=0)  # 5x5 filter
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # Flatten
        x = x.view(-1, 2 * 43 * 43)        
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

    window_width = 20

    X_train = rolling_window(np.array(X_train), window_width)
    X_train = np.transpose(X_train, (2, 3, 0, 1))
    r_train = r_train[:, window_width:]
    y_train = y_train[:, :, window_width:]
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = np.array(y_train)
    r_train = np.array(r_train)
    r_train = r_train.T
    r_train[r_train>0] = 1

    X_test = rolling_window(np.array(X_test), window_width)
    X_test = np.transpose(X_test, (2, 3, 0, 1))
    r_test = r_test[:, window_width:]
    y_test = y_test[:, :, window_width:]
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = np.array(y_test)
    r_test = np.array(r_test)
    r_test = r_test.T
    r_test[r_test>0] = 1    

    model = SimpleCNN()
    n_hidden = 100
    # Create a reservoir
    reservoir = Reservoir(units=n_hidden, lr=0.9, connectivity=0.1, input_scaling=1, spectral_radius=0.9)

    train_states = []
    
    for input in X_train:
        input = input.unsqueeze(1)
        input = model(input)
        input = input.detach().numpy()
        states = reservoir.run(input)
        states = states[-1]
        train_states.append(states)
        reservoir = reservoir.reset()
    train_states = np.vstack(train_states)

    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=42))
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(train_states, r_train)
    
    test_states = []
    for input in X_test:
        input = input.unsqueeze(1)
        input = model(input)
        input = input.detach().numpy()
        states = reservoir.run(input)
        states = states[-1]
        test_states.append(states)
        reservoir = reservoir.reset()
    test_states = np.vstack(test_states)    

    preds = clf.predict_proba(test_states)