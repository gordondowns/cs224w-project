
import sys

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
from collections import Counter

# **Model**

model_wavenumbers = np.load('data/processed/wavenumber_vals_v3.npy')
dataset_path = 'data/processed/v3.pt'
dataset = Crystals(dataset_path)

train_dataset, val_dataset, test_dataset = dataset.get_splits(deterministic=True)
train_counter = Counter()
val_counter = Counter()
test_counter = Counter()
for data in DataLoader(train_dataset, batch_size=1):
    train_counter[data.mineral[0]] += 1
for data in DataLoader(val_dataset, batch_size=1):
    val_counter[data.mineral[0]] += 1
for data in DataLoader(test_dataset, batch_size=1):
    test_counter[data.mineral[0]] += 1

print("train")
print(train_counter)
print("val")
print(val_counter)
print("test")
print(test_counter)
