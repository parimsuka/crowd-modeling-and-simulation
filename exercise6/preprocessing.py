"""
Preprocessing Methods

TODO: Finalize Docstring(s)

"""

import numpy as np
import numpy.typing as npt
import math
import scipy
import logging
import torch
from sklearn.preprocessing import MinMaxScaler


def get_ped_paths(ped_data):
    """
    Takes an Array of Pedestrian Data and divides it into multiple arrays, one for each pedestrian.
    Returns a list of these arrays.
    TODO: finalize Docstring

    :param ped_data:
    :return:
    """
    ped_ids = np.unique(ped_data[:, 0])
    ped_paths = []

    for i in range(ped_ids.size):
        ped_id = ped_ids[i]
        ped_mask = (np.isclose(ped_data[:, 0], ped_id))
        ped_paths.append(ped_data[ped_mask, :])

    return ped_paths


def get_ped_speeds_list(ped_paths):
    """
    Takes a list of pedestrian paths and returns a list of pedestrian speeds
    TODO: Finalize Docstring

    :param ped_paths:
    :return:
    """
    ped_speeds = []

    # Doing it in a for-loop first, should probably change that later
    for ped_path in ped_paths:
        # New Format: ID, Frame, Speed
        ped_speed = np.empty((ped_path.shape[0]-1, 3))

        for i in range(ped_speed.shape[0]):
            # Iterate over rows of single pedestrian
            ped_speed[i][0:2] = ped_path[i][0:2]  # Copy PedID and FrameID
            ped_speed[i][2] = np.linalg.norm(ped_path[i+1][2:5] - ped_path[i][2:5]) / (ped_path[i+1][1] - ped_path[i][1])

        ped_speeds.append(ped_speed)

    return ped_speeds


def get_ped_frames(ped_data):
    ped_frame_ids = np.unique(ped_data[:, 1])
    ped_frames = {}

    for i in range(ped_frame_ids.size):
        ped_id = ped_frame_ids[i]
        ped_mask = (np.isclose(ped_data[:, 1], ped_id))
        ped_frames[ped_id] = (ped_data[ped_mask, :])

    return ped_frames


def construct_kNN_tree(ped_data):
    # Compute kNN Tree for each Frame sequentially
    ped_frames = get_ped_frames(ped_data)
    kNN_frames = {}

    for key, frame in ped_frames.items():
        kNN_frames[key] = scipy.spatial.KDTree(frame[:, 0:5])

    return kNN_frames


def build_data_structure(trees_dict, k=10):
    ks = [k for k in range(2, k+2)]  # build the k's without [1] so that self isn't included

    returned_dict = {}

    for key, tree in trees_dict.items():

        # 1. Calculate the Input Data of the following Shape:
        # Length: 2K+1
        # mean_spacing - x_diff - y_diff - x_diff2 - y_diff 2 - ...
        # Note: Why split in x and y? Why not just distance?

        for i in range(tree.n):
            dict_key = (tree.data[i, 0], tree.data[i, 1])
            returned_dict[dict_key] = np.empty(2*len(ks)+1)
            # returns distances, indexes
            distances, indexes = tree.query(tree.data[i], k=ks)

            # calculate mean spacing
            returned_dict[dict_key][0] = np.mean(distances)

            # calculate y_diff and x_diff
            for j, k in enumerate(indexes):
                if k < tree.n:
                    returned_dict[dict_key][2*j+1] = tree.data[k, 2] - tree.data[i, 2]  # x_diff
                    returned_dict[dict_key][2*j+2] = tree.data[k, 3] - tree.data[i, 3]  # y_diff

                else:
                    logging.debug(f"Tree: {key}\tTreeIndex: {i}\tNeighborIndex: {k}\n"
                                  f"\tk: {len(ks)}\tTreeItems: {tree.n}\tConsideredTreeItems: {tree.n - 1}\n"  # -1 Item considered since self is removed
                                  f"\tNot enough neighbors, adding np.inf to result array.")

                    returned_dict[dict_key][2 * j + 1] = np.inf
                    returned_dict[dict_key][2 * j + 2] = np.inf

    return returned_dict


def get_ped_speeds(ped_paths):
    """
    Takes a list of pedestrian paths and returns a list of pedestrian speeds
    TODO: Finalize Docstring

    :param ped_paths:
    :return:
    """
    ped_speeds = {}

    for ped_path in ped_paths:
        range_helper = ped_path.shape[0]-1
        for i in range(range_helper):
            # Iterate over rows of single pedestrian
            dict_key = (ped_path[i, 0], ped_path[i, 1])
            ped_speeds[dict_key] = np.linalg.norm(ped_path[i+1][2:5] - ped_path[i][2:5]) / (ped_path[i+1][1] - ped_path[i][1])

    return ped_speeds


def clean_dataset(dataset: list[dict]) -> (list[dict], int):
    """Removes all nan values from the dict"""

    logging.info("Cleaning data.")

    cleaned_dataset = []

    removed_item_counter = 0

    for listitem in dataset:
        if np.isfinite(listitem['distances']).all() and np.isfinite(listitem['speed']).all():
            cleaned_dataset.append(listitem)
        else:
            removed_item_counter += 1
            # print(f"\tBad Listitem: {listitem}")

    logging.info(f"Finished cleaning data, {removed_item_counter} items removed.")

    return cleaned_dataset, removed_item_counter


def merge_distance_speed(distance_values: dict,
                         speed_values: dict,
                         k: int,
                         hide_id_frame: bool = True
                         ) -> list[dict]:
    logging.info("Merging preprocessed data.")
    data_list = []

    for key, input_val in distance_values.items():
        data_item = {
            'distances': np.empty((2*k + 1)),
            'speed': np.empty(1),
        }
        if not hide_id_frame:
            data_item['id+frame'] = key
        try:
            data_item['speed'] = speed_values[key]
            data_item['distances'] = input_val
            data_list.append(data_item)
        except KeyError:
            logging.debug(f"\t\t{key} has no speed value.")
            continue

    for key, speed_val in speed_values.items():
        try:
            _ = distance_values[key]
        except KeyError:
            logging.debug(f"\t\t{key} has no distance value.")
            continue

    return data_list


def do_preprocessing(data: npt.ArrayLike,
                     k: int,
                     clean_data: bool = True,
                     hide_id_frame: bool = True
                     ) -> list[dict]:
    """Gets data, returns complete list of dicts"""

    # logging.basicConfig(level=logging.INFO)
    # logging.warning("Testwarning")

    # 1. get Input Values per ID+Frame
    logging.info("Preprocessing distance values.")
    trees_dict = construct_kNN_tree(data)
    distance_values = build_data_structure(trees_dict, k=k)

    # 2. get Speed Values per ID+Frame
    logging.info("Preprocessing speed values.")
    ped_paths = get_ped_paths(data)
    speed_values = get_ped_speeds(ped_paths)

    # 3. Build final list of items (dicts)
    data = merge_distance_speed(distance_values, speed_values, k, hide_id_frame)

    if clean_data:
        cleaned_data, removed_item_counter = clean_dataset(data)
        return cleaned_data
    else:
        return data

    
def normalize_data(dataset):
    
    scaler = MinMaxScaler()
    
    # Normalize data
    distances = torch.tensor(np.array([sample['distances'] for sample in dataset]))
    speed = torch.tensor(np.array([sample['speed'] for sample in dataset]))

    # Applying data normalization
    X = scaler.fit_transform(distances.T).T
    y = scaler.fit_transform(speed.reshape(-1, 1)).flatten()

    # Create a list of dictionaries
    data = []
    for i in range(len(X)):
        sample = {'distances': X[i], 'speed': y[i]}
        data.append(sample)

    return data


def prepare_weidmann_data(train_dataset, test_dataset): 
    """
    Takes a list of train and test dataset and converts to [mean spacing, speed] for each data point.

    :param ped_paths: list of pedestrian paths
    :return: train_x, train_y, test_x, test_y
    """
    
    train_x, train_y, test_x, test_y = [], [], [], []

    # Get only mean spacing and speed for training 
    for k in range(len(train_dataset)):
        for i in range (len(train_dataset[k])):
            distances = train_dataset[k][i]['distances']
            speed = train_dataset[k][i]['speed']
            distance = 0
            for j in range(1, 21, 2):
                distance += math.sqrt((distances[j])**2 + (distances[j+1])**2)
            train_x.append(distance/10)
            train_y.append(speed)

    # Get only mean spacing and speed for testing 
    for i in range (len(test_dataset)):
        distances = test_dataset[i]['distances']
        speed = test_dataset[i]['speed']
        distance = 0
        for j in range(1, 21, 2):
            distance += math.sqrt((distances[j])**2 + (distances[j+1])**2)
        test_x.append(distance/10)
        test_y.append(speed)

    return train_x, train_y, test_x, test_y




'''
import math
mean_spaces = []
for i in range(len(bottleneck_train_val_dataset[0])):
        #print(c_015_train_val_datasets[0][i])
        distance = 0
        for j in range(1, 21, 2):
            distance += math.sqrt((bottleneck_train_val_dataset[0][i]['distances'][j])**2 + (bottleneck_train_val_dataset[0][i]['distances'][j+1])**2)

        mean_spaces.append(distance/10)

print(mean_spaces)
print(bottleneck_train_val_dataset[i]['distances'][0])
'''