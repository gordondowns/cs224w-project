# auto-reload changed source files when they are imported

# add top repo dir to path so that src can be imported
import sys

from torch.nn.modules.activation import Sigmoid
sys.path.append("..")
sys.path.append('C:/Users/gordon/Desktop/cs224w-project')


import os.path as osp

import torch
from torch.nn import Linear, ReLU, MSELoss, Tanh

import numpy as np
from torch_geometric.nn import SchNet, Sequential
from torch_geometric.loader import DataLoader
from src.data.dataset import Crystals
from src.visualization.plotting import plot_spectra

# **Model**

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: Dimensions for lin layers
model = Sequential(
    'z, pos, batch',
    [
        (SchNet(),'z, pos, batch-> z'),
        Sigmoid(),
        # Linear(),
        # ReLU(inplace=True),
        # Linear()
    ]
).to(device)

# **Init**

# %%
save_path = 'data/processed/v1.pt'
dataset = Crystals(save_path)
dataset[0] 

# %%
train_dataset, val_dataset, test_dataset = dataset.get_splits(deterministic=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)#, weight_decay=5e-4)
loss_fn = MSELoss()

loader = DataLoader(train_dataset, batch_size=256)

# %% [markdown]
# **Train loop**

# %%
loss_list = []
model.train()
for epoch in range(500):
    print(epoch)
    optimizer.zero_grad()
    for i,data in enumerate(loader):
        # print(i)
        data = data.to(device)
        pred = model(data.z, data.pos, data.batch)

        # TODO: make sure pred and data.y have same dim and have corresponding elements
        loss = loss_fn(pred, data.y)
        loss_list.append(loss.item())
        # print(loss)
        loss.backward()
        optimizer.step()
        # print(pred)
        # print(pred.shape)

from matplotlib import pyplot as plt
plt.plot(loss_list,linewidth=.2)
plt.yscale('log')
plt.show()

model.eval()
for data in loader:
    data = data.to(device)
    pred = model(data.z, data.pos, data.batch).detach().cpu().numpy().flatten()
    true = data.y.detach().cpu().numpy().flatten()

    xs = np.arange(1000.0,500.0,-10.0)

    P = [(x, p) for (x, p) in zip(xs, pred)]
    Y = [(x, y) for (x, y) in zip(xs, true)]

    plot_spectra(P, Y)

    


