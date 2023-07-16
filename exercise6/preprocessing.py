"""
This module implements preprocessing methods that prepare the data from the dataset to be
used in the neural network.

author: Simon Blöchinger
"""

import numpy as np
import numpy.typing as npt
import math
import scipy
import logging
import torch
import typing as t
from sklearn.preprocessing import MinMaxScaler


def get_ped_paths(ped_data: npt.ArrayLike) -> list:
    """
    Takes the original array of pedestrian data and divides it into multiple lists for each pedestrian.
    Each list item then represents the path of one pedestrian.

    :param ped_data: The original location data of all pedestrians at each point in time.
    :return: A list of the locations grouped per pedestrian, representing the path a pedestrian took.
    """
    ped_ids = np.unique(ped_data[:, 0])
    ped_paths = []

    for i in range(ped_ids.size):
        ped_id = ped_ids[i]
        ped_mask = (np.isclose(ped_data[:, 0], ped_id))
        ped_paths.append(ped_data[ped_mask, :])

    return ped_paths


def get_ped_speeds(ped_paths: list, return_list: bool = False) -> t.Union[dict, list]:
    """
    Takes a list of pedestrian paths and returns the speed of the pedestrians.
    Can either return the speeds in a dictionary with the PedID and FrameID as key (default)
    or return the speeds in a list of lists for easier plotting of the speed behavior of one pedestrian.

    :param ped_paths: List of pedestrian locations grouped by pedestrain (the paths of each pedestrian).
    :param return_list: If false (default), returns the data in a dictionary. If true, returns the data in a list.
    :return: The speeds of the pedestrians in either a dictionary (key: (PedID, FrameID)) or a list of lists.
    """
    # Initialize the returned objects for each case (dictionary or list)
    if return_list:
        ped_speeds_list = []
    else:
        ped_speeds_dict = {}

    for ped_path in ped_paths:
        # Iterate over all pedestrian paths

        # The length of the returned speed values will be the length of the location values - 1
        ped_speed_len = ped_path.shape[0] - 1

        if return_list:
            ped_speed_listitem = np.empty((ped_speed_len, 3))

        for i in range(ped_speed_len):
            # Iterate over rows of single pedestrian
            if return_list:
                ped_speed_listitem[i][0:2] = ped_path[i][0:2]  # Copy PedID and FrameID
                ped_speed_listitem[i][2] = np.linalg.norm(ped_path[i + 1][2:5] - ped_path[i][2:5]) / (ped_path[i + 1][1] - ped_path[i][1])
            else:
                dict_key = (ped_path[i, 0], ped_path[i, 1])
                ped_speeds_dict[dict_key] = np.linalg.norm(ped_path[i+1][2:5] - ped_path[i][2:5]) / (ped_path[i+1][1] - ped_path[i][1])

        if return_list:
            ped_speeds_list.append(ped_speed_listitem)

    return ped_speeds_list if return_list else ped_speeds_dict


# def get_ped_speeds_list(ped_paths):
#     """
#     Takes a list of pedestrian paths and returns a list of pedestrian speeds
#     TODO: Finalize Docstring
#     TODO: remove duplicate function and put both in one
#
#     :param ped_paths:
#     :return:
#     """
#     ped_speeds = []
#
#     # Doing it in a for-loop first, should probably change that later
#     for ped_path in ped_paths:
#         # New Format: ID, Frame, Speed
#         ped_speed = np.empty((ped_path.shape[0]-1, 3))
#
#         for i in range(ped_speed.shape[0]):
#             # Iterate over rows of single pedestrian
#             ped_speed[i][0:2] = ped_path[i][0:2]  # Copy PedID and FrameID
#             ped_speed[i][2] = np.linalg.norm(ped_path[i+1][2:5] - ped_path[i][2:5]) / (ped_path[i+1][1] - ped_path[i][1])
#
#         ped_speeds.append(ped_speed)
#
#     return ped_speeds


def get_ped_frames(ped_data: npt.ArrayLike) -> dict:
    """
    Takes the original array of pedestrian data and stores it in a dictionary divided by the FrameID (used as key).
    Each dictionary item then represents the state of all pedestrians in a frame.

    :param ped_data: The original location data of all pedestrians at each point in time.
    :return: A dictionary of the locations grouped per frame, representing the state of a frame.
    """
    # Get a list of all FrameIDs
    frame_ids = np.unique(ped_data[:, 1])
    frames = {}

    for i in range(frame_ids.size):
        # Iterate through all frames in the data
        ped_id = frame_ids[i]

        # Prepare a mask for gathering all data that have the same FrameID value as our current frame
        ped_mask = (np.isclose(ped_data[:, 1], ped_id))

        # Store all data that corresponds to our current frame at the correct dictionary key
        frames[ped_id] = (ped_data[ped_mask, :])

    return frames


def construct_kNN_tree(ped_data: npt.ArrayLike) -> dict:
    """
    Constructs multiple KDTrees using the SciPy library for the pedestrian location data.
    This tree structure allows for quick querying of the k nearest neighbors of a location.
    First, divides the data into each frame.
    Then, constructs a separate KDTree for each frame to query the k nearest neighbors for each frame.

    :param ped_data: The original location data of all pedestrians at each point in time.
    :return: Dictionary containing KDTrees for each frame.
    """
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


def prepare_weidmann_data(train_dataset, test_dataset, k): 
    """
    Takes a list of train and test dataset and converts to [mean spacing, speed] for each data point.

    :param train_dataset: list of train dataset
    :param test_dataset: list of test dataset

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
            #Adjust distance to meters and speed to m/s
            train_x.append(distance/(100*k))
            train_y.append(speed/10)

    # Get only mean spacing and speed for testing 
    for i in range (len(test_dataset)):
        distances = test_dataset[i]['distances']
        speed = test_dataset[i]['speed']
        distance = 0
        for j in range(1, 21, 2):
            distance += math.sqrt((distances[j])**2 + (distances[j+1])**2)
        #Adjust distance to meters and speed to m/s
        test_x.append(distance/(100*k))
        test_y.append(speed/10)

    return train_x, train_y, test_x, test_y
