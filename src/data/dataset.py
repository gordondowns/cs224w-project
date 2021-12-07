from typing import Optional, Callable, List

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import (InMemoryDatset)

class Crystals(InMemoryDatset):
    def __init__(
        self, 
        root: str, 
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    def get_splits(self, deterministic=True):
        seed = 0 if deterministic else None
        train, val_test = train_test_split(self.data, test_size=.3, random_state=seed)
        val, test = train_test_split(val_test, test_size=.5, random_state=seed)
        return train, val, test
