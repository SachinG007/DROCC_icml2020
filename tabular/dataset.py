import torch
import numpy as np
import torch.utils.data as utils
from torch.utils.data import Sampler, Dataset

"""
Function to create pytorch dataset from numpy arrays 
X is n times d, y is n times 1
"""
class CustomDataset(Dataset):
    def __init__(self,
                 X,
                 y):
        self.data=X
        self.targets=y
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.data[idx]), (self.targets[idx]), torch.tensor([0])

