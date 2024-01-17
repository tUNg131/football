import os
import re
import json
import h5py
import torch
import random
import numpy as np
import multiprocessing

from functools import reduce
from collections import deque

from torch.utils.data import Dataset

BALL_DIR = "simulated.samples.ball"
CENTROID_DIR = "simulated.samples.centroids"
JOINT_DIR = "simulated.samples.joints"
JOINT_KEYS = [
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


class HumanPoseDataset(Dataset):
    def __init__(self, data_directory, timesteps=32, delta_t=0.25):
        if os.path.isfile(data_directory):
            with h5py.File(data_directory, "r") as f:
                ds = f['data']
                self._data = np.empty(ds.shape, dtype=np.float32)
                ds.read_direct(self._data)
                print(f"Loaded {data_directory}")
            return

        self.data_directory = data_directory
        self.timesteps = timesteps
        self.delta_t = delta_t
        self._data = None

        self.init_dataset()


    def init_dataset(self, force=False):
        if self._data and not force:
            return
        
        data = deque()
        for match_fullpath in self.get_match_fullpaths():
            with multiprocessing.Pool() as p:
                results = p.map(self.process_minute_data,
                                self.get_minute_data(match_fullpath))

            match_data = reduce(
                lambda a, b: {**a, **b}, (r[1] for r in results))

            ball_in_play = set().union(*(result[0] for result in results))

            players = set().union(*(result[2] for result in results))

            timestamps = sorted(ball_in_play)
            for player_id in players:
                sequence = []
                last_played = 0
                for t in timestamps:
                    joints_data = self.get_joints_data(match_data, player_id, t)
                    if t - last_played > self.delta_t or joints_data is None:
                        sequence = []
                    else:
                        sequence.append(joints_data)
                        if len(sequence) >= self.timesteps:

                            data.append(sequence)

                            sequence = []
                    last_played = t

            # Logging
            print(f"Loaded {match_fullpath}")
        
        # prevent memory leak
        self._data = np.array(data, dtype="f")


    def process_minute_data(self, data):
        ball_data, centroid_data, joint_data, minute = data

        play = set()
        players = set()
        match = {}

        for ball in ball_data:
            second = ball['time']
            if ball['play'] == "In":
                time = self.convert_to_second(minute, second)
                play.add(time)

        for player in joint_data:
            joint_coordinates = player['joints'][0]

            second = joint_coordinates['time']
            time = self.convert_to_second(minute, second)

            id = player['trackId']
            if time in play:
                for joint in JOINT_KEYS:
                    match[id, time, joint] = joint_coordinates[joint]

            players.add(id)

        for player in centroid_data:
            centroid_coordinates = player['centroid'][0]

            second = centroid_coordinates['time']
            time = self.convert_to_second(minute, second)

            if time in play:
                id = player['trackId']
                match[id, time, 'centroid'] = centroid_coordinates['pos']
        
        return play, match, players


    def get_minute_data(self, match_fullpath):
        ball_fullpath = os.path.join(match_fullpath, BALL_DIR)
        for filename in os.listdir(ball_fullpath):
            minute = self.get_minute_from_filename(filename)

            if minute is None:
                continue

            ball_data = self.get_data(
                os.path.join(match_fullpath, BALL_DIR, filename))

            prefix = filename[:-5] # remove the "_ball"
            centroid_data = self.get_data(
                os.path.join(
                    match_fullpath, CENTROID_DIR, prefix + "_centroids")
            )

            joint_data = self.get_data(
                os.path.join(
                    match_fullpath, JOINT_DIR, prefix + "_joints")
            )

            yield (
                ball_data['samples']['ball'],
                centroid_data['samples']['people'],
                joint_data['samples']['people'],
                minute
            )

    
    def get_match_fullpaths(self):
        fullpaths = []
        for entry in os.listdir(self.data_directory):
            path = os.path.join(self.data_directory, entry)

            if entry.startswith("."):
                continue

            if not os.path.isdir(path):
                continue
            
            fullpaths.append(path)
        return fullpaths


    @staticmethod
    def get_joints_data(match, id, t):
        data = []
        for joint in JOINT_KEYS + ['centroid']:
            k = id, t, joint
            joint_data = match.get(k)
            if joint_data is None:
                return
            data.append(joint_data)
        return data


    @staticmethod
    def convert_to_second(minute, second):
        return float(minute) * 60 + float(second)


    @staticmethod
    def get_data(path):
        with open(path, "r") as f:
            content = json.load(f)
        return content


    @staticmethod
    def get_minute_from_filename(filename):
        m = re.search(r'_(1|2)_(\d{1,2})(?:_(\d{1,2}))?_football_', filename)

        if not m:
            return

        t = 0 if m.group(1) == "1" else 55

        t += int(m.group(2)) - 1

        if m.group(3):
            t += int(m.group(3))

        return t


    @staticmethod
    def drop(data, max_gap_size=3):
        sequence_length = data.size(dim=0)

        assert sequence_length - max_gap_size - 2 >= 1

        gap_size = random.randint(1, max_gap_size)
        start_index = random.randint(1, sequence_length - gap_size - 2)

        data[start_index:start_index + gap_size] = float('nan')

        return data


    def preprocessing(self, data):
        # Subtract mid-hip
        data = data - data[..., 13, np.newaxis, :]

        # Remove mid-hip
        data = np.delete(data, 13, axis=1)

        return data


    def __getitem__(self, idx):
        data = self.preprocessing(self._data[idx])

        # rank = torch.distributed.get_rank()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = torch.tensor(data, dtype=torch.float, device=device)

        target = data.clone()

        data = self.drop(data)

        return data, target


    def __len__(self):
        return len(self._data)

    def save(self, filepath):
        with h5py.File(filepath, "w") as f:
            f.create_dataset("data", data=self._data, dtype="float32")
        print(f"Saved to {filepath}")