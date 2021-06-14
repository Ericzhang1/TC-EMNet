import torch
import os
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from random import Random

class disease_progression(Dataset):
    def __init__(self, data_dir, args):
        self.directory = data_dir
        if args.data_type == 0:
            self.data = np.load(f'{self.directory}/data.npz')
        elif args.data_type == 1:
            self.data = np.load(f'{self.directory}/ppmi_data_clean.npz')
        else:
            print('wrong data choice')
            exit(1)
        
        self.x = self.data['data_x']
        self.y = self.data['data_y']
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.int)
        self.length = np.asarray([np.sum(v) for v in self.y])
        self.length = self.length.astype(np.int)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'patient': self.x[idx], 'label': self.y[idx], 'length': self.length[idx]}
        return sample

class disease_progression_aux(Dataset):
    def __init__(self, dataset: disease_progression, train: bool, valid: bool, test: bool, seed: int = 0, fold: int = 0):
        assert train + valid + test == 1
        
        self.random = Random(seed)
        
        idx = np.arange(len(dataset)).tolist()
        self.random.shuffle(idx)

        start, end = int(fold * 0.2 * len(idx)), int((fold + 1) * 0.2 * len(idx))
        test_indices = idx[start : end]
        train_indices = idx[:start] + idx[end:]
        if train:
            indices = train_indices[:int(0.9 * len(train_indices))]
        elif valid:
            indices = train_indices[int(0.9 * len(train_indices)):]
        else:
            indices = test_indices
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return (len(self.indices))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.dataset[self.indices[idx]]

if __name__ == '__main__':
    dataset = disease_progression(data_dir='./data', data=0)
    