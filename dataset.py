import os
import re
import json
import h5py
import torch
import random

from collections import defaultdict

from torch.utils.data import Dataset, IterableDataset

BALL_DIR = "simulated.samples.ball"
CENTROID_DIR = "simulated.samples.centroids"
JOINTS_DIR = "simulated.samples.joints"
LABELS = [
    "lAnkle",
    "lBigToe",
    "lEar",
    "lElbow",
    "lEye",
    "lHeel",
    "lHip",
    "lKnee",
    "lPinky",
    "lShoulder",
    "lSmallToe",
    "lThumb",
    "lWrist",
    "midHip",
    "neck",
    "nose",
    "rAnkle",
    "rBigToe",
    "rEar",
    "rElbow",
    "rEye",
    "rHeel",
    "rHip",
    "rKnee",
    "rPinky",
    "rShoulder",
    "rSmallToe",
    "rThumb",
    "rWrist"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HumanPoseIterableDataset(IterableDataset):
    def __init__(self, data_directory, timesteps=32, frame_rate=4):
        self.data_directory = data_directory
        self.timesteps = timesteps
        self.frame_rate = frame_rate

    def __iter__(self):
        for match_path in self.get_match_fullpaths():
            sequences = defaultdict(list)
            last_t = defaultdict(int)

            for file_paths, minute in self.get_files(match_path):
                ball_path, centroids_path, joints_path = file_paths

                play = set()
                minute_data = defaultdict(dict)

                with open(ball_path, "r") as ball_file, \
                     open(centroids_path, "r") as centroid_file, \
                     open(joints_path, "r") as joints_file:
                
                    ball_json = json.load(ball_file)
                    for b in ball_json['samples']['ball']:
                        time = float(b['time'])
                        if b['play'] == "In":
                            play.add(time)

                    joints_json = json.load(joints_file)
                    for pp in joints_json['samples']['people']:
                        time = float(pp['joints'][0]['time'])
                        if time in play:
                            minute_data[pp['trackId']][time] = [pp['joints'][0][l] for l in LABELS]

                    centroid_json = json.load(centroid_file)
                    for pp in centroid_json['samples']['people']:
                        time = float(pp['centroid'][0]['time'])
                        if time in play:
                            minute_data[pp['trackId']][time].append(pp['centroid'][0]['pos'])

                    T = sorted(play)

                    for player in minute_data.keys():
                        for t in T:
                            if minute + t - last_t[player] == (1 / self.frame_rate) and t in minute_data[player]:
                                sequences[player].append(minute_data[player][t])
                                if len(sequences[player]) >= self.timesteps:

                                    data, target = self.prepare(sequences[player])
                                    yield data, target

                                    sequences[player] = []
                                else:
                                    sequences[player] = []

                                last_t[player] = minute + t

    def get_key(self, filename):
        m = re.search(r'_(1|2)_(\d{1,2})(?:_(\d{1,2}))?_football_', filename)

        if not m:
            return

        t = 0 if m.group(1) == "1" else 55

        t += int(m.group(2)) - 1

        if m.group(3):
            t += int(m.group(3))

        return t

    def get_files(self, path):
        files = []
        ball_path = os.path.join(path, BALL_DIR)
        for filename in os.listdir(ball_path):
            t = self.get_key(filename)

            if t is None:
                continue

            prefix = filename.replace("_ball", "_")

            paths = (
                os.path.join(path, BALL_DIR, prefix + "ball"),
                os.path.join(path, CENTROID_DIR, prefix + "centroids"),
                os.path.join(path, JOINTS_DIR, prefix + "joints")
            )

            files.append((paths, t))
        files.sort(key=lambda x: x[1])
        return files

    def get_match_fullpaths(self):
        fullpaths = []
        for entry in os.listdir(self.data_directory):
            path = os.path.join(self.data_directory, entry)

            if entry.startswith("."):
                continue

            if not os.path.isdir(path):
                continue
            
            fullpaths.append(path)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None: # single-process
            return fullpaths
        else:
            worker_fullpaths = []
            worker_id = worker_info.id
            for i, path in enumerate(fullpaths):
                if i % worker_id.num_workers == worker_id:
                    worker_fullpaths.append(path)
            return worker_fullpaths
        
    def drop(self, data, max_gap_size=3):
        sequence_length = data.size(dim=0)

        assert sequence_length - max_gap_size - 2 >= 1

        gap_size = random.randint(1, max_gap_size)
        start_index = random.randint(1, sequence_length - gap_size - 2)

        data[start_index:start_index + gap_size] = float('nan')

        return data
    
    def prepare(self, seq):
        data = torch.tensor(seq, dtype=torch.float, device=device)

        target = data.clone()

        data = self.drop(data)

        return data, target


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