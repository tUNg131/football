import os

import torch
import h5py
import numpy as np


class HumanPoseDataset(torch.utils.data.Dataset):
    def __init__(self, path, device=None):
        if not os.path.isfile(path):
            raise

        self.path = path
        self.device = device

        with h5py.File(path, "r") as f:
            ds = f['data']
            self._data_array = np.empty(ds.shape, dtype=np.float32)
            ds.read_direct(self._data_array)


    def preprocessing(self, data):
        # Subtract mid-hip
        data = data - data[..., 13, np.newaxis, :]

        # Remove joints
        removed_indexes = [1, 2, 4, 5, 8, 10, 11, 13, 17, 18, 20, 21, 24, 26, 27]
        data = np.delete(data, removed_indexes, axis=1)

        return data


    def __getitem__(self, idx):
        raw = self._data_array[idx]
        clean = self.preprocessing(raw)
        return torch.tensor(clean, dtype=torch.float, device=self.device)


    def __len__(self):
        return len(self._data_array)
