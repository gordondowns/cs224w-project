# auto-reload changed source files when they are imported

# add top repo dir to path so that src can be imported
import sys
sys.path.append("..")
sys.path.append('C:/Users/gordon/Desktop/cs224w-project')


import os.path as osp

import torch
from torch.nn import Linear, ReLU, MSELoss

from torch_geometric.nn import SchNet, Sequential
from torch_geometric.loader import DataLoader
from src.data.dataset import Crystals

# **Model**

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: Dimensions for lin layers
model = Sequential(
    'z, pos',
    [
        (SchNet(),'z, pos -> z'),
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
train_dataset, val_dataset, test_dataset = dataset.get_splits(deterministic=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)
loss_fn = MSELoss()

loader = DataLoader(train_dataset, batch_size=1)

# %% [markdown]
# **Train loop**

# %%
loss_list = []
model.train()
for epoch in range(5):
    print(epoch)
    optimizer.zero_grad()
    for i,data in enumerate(loader):
        # print(i)
        data = data.to(device)
        pred = model(data.z, data.pos)

        # TODO: make sure pred and data.y have same dim and have corresponding elements
        loss = loss_fn(pred, data.y)
        loss_list.append(loss.item())
        # print(loss)
        loss.backward()
        optimizer.step()
        # print(pred)
        # print(pred.shape)

from matplotlib import pyplot as plt
plt.plot(loss_list)
plt.yscale('log')
plt.show()



