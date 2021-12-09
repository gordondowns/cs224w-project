# auto-reload changed source files when they are imported

# add top repo dir to path so that src can be imported
import sys
import os
from datetime import datetime
import copy

from torch.utils.tensorboard import SummaryWriter
sys.path.append("..")
sys.path.append('C:/Users/gordon/Desktop/cs224w-project')


import torch
from torch.nn import Linear, ReLU, MSELoss, Dropout

import numpy as np
from torch_geometric.nn import Sequential#, SchNet
from torch_geometric.loader import DataLoader
from src.models.schnet import SchNet
from src.data.dataset import Crystals
from src.visualization.plotting import plot_spectra
from matplotlib import pyplot as plt
# **Model**

model_wavenumbers = np.load('data/processed/wavenumber_vals_v3.npy')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Sequential(
    'z, pos, batch',
    [
        (SchNet(hidden_channels=1024,out_features=256),'z, pos, batch->z'),
        ReLU(inplace=True),
        Dropout(0.5),
        Linear(256, 128),
        ReLU(inplace=True),
        Dropout(0.5),
        Linear(128, len(model_wavenumbers)),
    ]
).to(device)
print(model)
dataset_path = 'data/processed/v3.pt'
dataset = Crystals(dataset_path)

train_dataset, val_dataset, test_dataset = dataset.get_splits(deterministic=True)

# load best model
state_dict = torch.load('models/20211209-121703_epoch0000_mse0.4646.pt')
model.load_state_dict(state_dict)

model.eval()
for data in DataLoader(test_dataset, batch_size=1):
    data = data.to(device)
    pred = model(data.z, data.pos, data.batch).detach().cpu().numpy().flatten()
    true = data.y.detach().cpu().numpy().flatten()

    P = [(x, p) for (x, p) in zip(model_wavenumbers, pred)]
    Y = [(x, y) for (x, y) in zip(model_wavenumbers, true)]

    plot_spectra(P, Y, title=data.mineral[0])



