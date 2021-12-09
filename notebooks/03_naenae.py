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

# **Init**

log_dir = 'logs'
now = datetime.now()
dt_string = now.strftime("%Y%m%d-%H%M%S")
log_path = os.path.join(log_dir, dt_string)
writer = SummaryWriter(log_path)
dataset_path = 'data/processed/v3.pt'
dataset = Crystals(dataset_path)
save_model_dir = 'models'

train_dataset, val_dataset, test_dataset = dataset.get_splits(deterministic=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
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
save_model_at_most_every_n_epochs = 50
best_val_mse_epoch = 0
best_model_wts = copy.deepcopy(model.state_dict())
best_val_mse = np.inf
for epoch in range(30001):
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

    if epoch % save_model_at_most_every_n_epochs == 0:
        if val_mse < best_val_mse:
            best_model_wts = copy.deepcopy(model.state_dict())
            model_save_path = os.path.join(save_model_dir, f'{dt_string}_epoch{epoch:04d}_mse{val_mse:.4f}.pt')
            torch.save(model,model_save_path)
            print("validation MSE loss:",val_mse)
            print("model saved to",model_save_path)

# save final model, just out of curiosity
model_save_path = os.path.join(save_model_dir, f'{dt_string}_epoch{epoch:04d}_mse{val_mse:.4f}.pt')
torch.save(model,model_save_path)
print("terminal validation MSE loss:",val_mse)
print("terminal model saved to",model_save_path)

# load best model
model.load_state_dict(best_model_wts)

plt.plot(loss_list,linewidth=.2)
plt.yscale('log')
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.show()

model.eval()
for data in DataLoader(val_dataset, batch_size=1):
    data = data.to(device)
    pred = model(data.z, data.pos, data.batch).detach().cpu().numpy().flatten()
    true = data.y.detach().cpu().numpy().flatten()

    P = [(x, p) for (x, p) in zip(model_wavenumbers, pred)]
    Y = [(x, y) for (x, y) in zip(model_wavenumbers, true)]

    plot_spectra(P, Y, title=data.mineral[0])



