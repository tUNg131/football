import os
import re

import orjson
import numpy as np


BALL_DIRECTORY = "simulated.samples.ball"
JOINTS_DIRECTORY = "simulated.samples.joints"
CENTROIDS_DIRECTORY = "simulated.samples.centroids"
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
EXCLUDED_FILES = [
    ".DS_Store"
]


class FileType:
    BALL = 1
    JOINTS = 2
    CENTROIDS = 3


def get_time(minute, second):
    return int((float(minute) * 6000 + round(float(second) * 100)))


def valid_file(file):
    if file in EXCLUDED_FILES:
        return False
    return True


def load_json(path):
    with open(path, "rb") as f:
        bytes = f.read()
        contents = orjson.loads(bytes)
        return contents, path


def list_files_in_folder(folder_path):
    files_list = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not valid_file(file):
                continue
            
            file_path = os.path.join(root, file)
            files_list.append(file_path)

    return files_list


def get_minute_from_filename(filename):
    m = re.search(r'_(1|2)_(\d{1,2})(?:_(\d{1,2}))?_football_', filename)
    return (-1 if m.group(1) == "1" else 59) + int(m.group(2)) + (int(m.group(3)) if m.group(3) else 0)


def get_file_type_from_directory(directory):
    if directory == BALL_DIRECTORY:
        file_type = FileType.BALL
    elif directory == CENTROIDS_DIRECTORY:
        file_type = FileType.CENTROIDS
    elif directory == JOINTS_DIRECTORY:
        file_type = FileType.JOINTS
    return file_type


def get_file_info(file_path):
    directory, filename = os.path.split(file_path)
    _, directory = os.path.split(directory)

    return (
        get_file_type_from_directory(directory),
        get_minute_from_filename(filename)
    )


def get_joints_data(match_data, player_id, time):
    data = []
    for joint in JOINT_KEYS + ['centroid']:
        d = match_data.get((player_id, time, joint), None)
        if d is None:
            return
        data.append(d)
    return data


def load_match(path):
    match_data = dict()
    ball_in_play = set()
    available_players = set()

    for path in list_files_in_folder(path):
        contents, path = load_json(path)

        file_type, minute = get_file_info(path)

        if file_type == FileType.BALL:
            for sample in contents['samples']['ball']:
                second = sample['time']
                if sample['play'] == "In":
                    time = get_time(minute, second)
                    ball_in_play.add(time)

        elif file_type == FileType.CENTROIDS:
            for sample in contents['samples']['people']:
                data = sample['centroid'][0]
                second = data['time']
                time = get_time(minute, second)
                player_id = sample['trackId']
                match_data[player_id, time, 'centroid'] = data['pos']
                available_players.add(player_id)

        elif file_type == FileType.JOINTS:
            for sample in contents['samples']['people']:
                data = sample['joints'][0]
                second = data['time']
                time = get_time(minute, second)
                player_id = sample['trackId']
                for joint in JOINT_KEYS:
                    match_data[player_id, time, joint] = data[joint]
                available_players.add(player_id)

    return match_data, ball_in_play, available_players


def list_subfolders(root):
    subfolders = []
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        subfolders.append(path)
    return subfolders


def data_generator(root, ball_delta_t, joints_delta_t, sequence_length):
    for path in list_subfolders(root):
        match_data, ball_in_play, available_players = load_match(path)

        timestamps = sorted(ball_in_play)
        for player_id in available_players:
            sequence = []
            last_played = 0
            joints_time = 0
            for ball_time in timestamps:
                if ball_time - last_played > ball_delta_t:
                    sequence = []
                    joints_time = ((ball_time - 1) // joints_delta_t + 1) * joints_delta_t
                else:
                    while joints_time <= ball_time:
                        joints_data = get_joints_data(match_data, player_id, joints_time)
                        if joints_data is None:
                            sequence = []
                        else:
                            sequence.append(joints_data)
                            if len(sequence) >= sequence_length:
                                
                                yield sequence

                                sequence = []
                        joints_time += joints_delta_t
                last_played = ball_time


def load_data_array_from(root, ball_delta_t, joints_delta_t, sequence_length):
    data_array = np.fromiter(data_generator(root, ball_delta_t, joints_delta_t, sequence_length),
                             dtype=np.dtype((float, (sequence_length, 30, 3))))
    return data_array