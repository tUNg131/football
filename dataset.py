import os
import re
import h5py
import torch
import random
from torch.utils.data import Dataset


class HumanPoseDataset(Dataset):
    def __init__(self, root, timesteps=32):
        self.dataset = []

        for entry in os.listdir(root):
            filename = os.path.join(root, entry)

            # Guard against non-h5 files
            if not (os.path.isfile(filename) and entry.endswith(".h5")):
                continue

            with h5py.File(filename, 'r') as h5_file:
                for key in h5_file.keys():
                    m = re.search(r"d-(\d+)", key)
                    if not m:
                        continue

                    pose_dataset = h5_file[m.group(0)]
                    time_dataset = h5_file["t-" + m.group(1)]

                    raw = []
                    last_t = 0.
                    for p, t in zip(pose_dataset, time_dataset):
                        if t - last_t == 0.25:
                            raw.append(p)
                            if len(raw) >= timesteps:

                                self.dataset.append(raw)

                                raw = []
                        else:
                            raw = []
                        last_t = t

    def drop(self, data, max_gap_size=3):
        sequence_length = data.size(dim=0)

        assert sequence_length - max_gap_size - 2 >= 1

        gap_size = random.randint(1, max_gap_size)
        start_index = random.randint(1, sequence_length - gap_size - 2)

        data[start_index:start_index + gap_size] = float('nan')

        return data
    
    def __getitem__(self, index):
        data = torch.tensor(self.dataset[index],
                            dtype=torch.float)
        target = data.clone()

        data = self.drop(data)

        return data, target