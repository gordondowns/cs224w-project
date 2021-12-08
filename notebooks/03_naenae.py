# auto-reload changed source files when they are imported

# add top repo dir to path so that src can be imported
import sys
import os
from datetime import datetime

from torch.nn.modules.activation import Sigmoid
from torch.utils.tensorboard import SummaryWriter
sys.path.append("..")
sys.path.append('C:/Users/gordon/Desktop/cs224w-project')


import os.path as osp

import torch
from torch.nn import Linear, ReLU, MSELoss, Tanh

import numpy as np
from torch_geometric.nn import SchNet, Sequential
from torch_geometric.loader import DataLoader, dataloader
from src.data.dataset import Crystals
from src.visualization.plotting import plot_spectra

# **Model**

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Sequential(
    'z, pos, batch',
    [
        (SchNet(hidden_channels=1024),'z, pos, batch->z'),
        # Linear(),
        ReLU(inplace=True),
        Linear(256, 128),
        ReLU(inplace=True),
        Linear(128, 50),
    ]
).to(device)

# **Init**

log_dir = 'logs'
now = datetime.now()
dt_string = now.strftime("%Y%m%d-%H%M%S")
log_path = os.path.join(log_dir, dt_string)
writer = SummaryWriter(log_path)
save_path = 'data/processed/v1.pt'
dataset = Crystals(save_path)
dataset[0] 

train_dataset, val_dataset, test_dataset = dataset.get_splits(deterministic=False)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)#, weight_decay=5e-4)
loss_fn = MSELoss()

train_loader = DataLoader(train_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)

# **Train loop**

def val(model, val_dataloader):
    model.eval()
    with torch.no_grad():
        mse = 0
        for data in val_dataloader:
            data = data.to(device)
            pred = model(data.z, data.pos, data.batch)

            mse += loss_fn(pred, data.y)
        return mse / len(val_dataloader)


loss_list = []
for epoch in range(1800):
    model.train()
    print(epoch)
    optimizer.zero_grad()
    for i,data in enumerate(train_loader):
        data = data.to(device)
        pred = model(data.z, data.pos, data.batch)

        loss = loss_fn(pred, data.y)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
    
    train_mse = val(model, train_loader)
    val_mse = val(model, val_loader)
    writer.add_scalar('train MSE', train_mse, epoch)
    writer.add_scalar('val MSE', val_mse, epoch)

from matplotlib import pyplot as plt
plt.plot(loss_list,linewidth=.2)
plt.yscale('log')
plt.show()

model.eval()
for data in DataLoader(val_dataset, batch_size=1):
    data = data.to(device)
    pred = model(data.z, data.pos, data.batch).detach().cpu().numpy().flatten()
    true = data.y.detach().cpu().numpy().flatten()

    xs = np.arange(1000.0,500.0,-10.0)

    P = [(x, p) for (x, p) in zip(xs, pred)]
    Y = [(x, y) for (x, y) in zip(xs, true)]

    plot_spectra(P, Y)

    


