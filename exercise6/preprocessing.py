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


def construct_kNN_trees(ped_data: npt.ArrayLike) -> dict:
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


def get_ped_distances(kNN_trees_dict: dict, k: int = 10) -> dict:
    """
    Generates the neccessary input values needed for the neural network.
    The values are:
    1. The mean distance of the k nearest neighbors
    2. The relative distances of the k nearest neighbors each
        These relative distances are split into the x- and y- direction
    In total, there are 2k+1 total input values with the following structure for each dict item:
        mean_spacing, x_diff_1, y_diff_1, x_diff_2, y_diff_2, ..., x_diff_k, y_diff_k

    :param kNN_trees_dict: Dictionary containing KDTrees for each frame.
    :param k: The number of nearest neighbors the KDTrees are queried for.
    :return: The distance values used as input for the neural network in a dictionary keyed by the PedID and FrameID.
    """
    # build the k's without [1] so that self isn't included
    # Querying the KDTree for a given pedestrian position will return the pedestrian itself as one of the closest neighbors
    # To query for k neighbors of a pedestrian without getting the pedestrian itself, we generate the following list:
    # [2, ..., k+1]
    # This list has k items but starts at 2 (skipping the closest nearest neighbor, which is the pedestrian itself)
    ks = [k for k in range(2, k+2)]

    returned_dict = {}

    for key, tree in kNN_trees_dict.items():
        # Iterate through all frames (saved in separate kNN Trees)

        for i in range(tree.n):
            # Iterate through all pedestrians present in a given frame
            # and calculate the nearest neighbors for all of them

            # The key for the kNN dictionary: (PedID, FrameID)
            dict_key = (tree.data[i, 0], tree.data[i, 1])

            # Initialize the returned dictionary item of length 2k+1
            returned_dict[dict_key] = np.empty(2*len(ks)+1)

            # Query the KDTree for the k nearest neighbors,
            #   returning their distances to the queried position and their indexes
            #   in the array containing all pedestrians in a given frame (tree.n)
            distances, indexes = tree.query(tree.data[i], k=ks)

            # Calculate the mean spacing and store it at index 0 of our returned numpy array
            returned_dict[dict_key][0] = np.mean(distances)

            # Calculate the y_diffs and x_diffs of the k nearest neighbors and store them at
            #   indexes 1..2k of our returned numpy array
            for j, k in enumerate(indexes):
                if k < tree.n:
                    returned_dict[dict_key][2 * j + 1] = tree.data[k, 2] - tree.data[i, 2]  # x_diff
                    returned_dict[dict_key][2 * j + 2] = tree.data[k, 3] - tree.data[i, 3]  # y_diff

                else:
                    # If the tree for the currently processed frame does not have enough pedestrians to
                    #   log k nearest neighbors, the missing pedestrians are added using np.inf as relative distances.
                    # These datapoints can later be removed using clean_dataset(), to allow the network
                    #   to train only with data where enough neighbors are available.
                    logging.debug(f"Tree: {key}\tTreeIndex: {i}\tNeighborIndex: {k}\n"
                                  f"\tk: {len(ks)}\tTreeItems: {tree.n}\tConsideredTreeItems: {tree.n - 1}\n"  # -1 Item considered since self is removed
                                  f"\tNot enough neighbors, adding np.inf to result array.")

                    returned_dict[dict_key][2 * j + 1] = np.inf
                    returned_dict[dict_key][2 * j + 2] = np.inf

    return returned_dict


def clean_dataset(dataset: list[dict]) -> (list[dict], int):
    """
    Removes data points from our dataset that contain a np.inf or np.nan value.

    :param dataset: The dataset from which to remove invalid datapoints from.
    :return: The cleaned dataset as well as the number of removed items.
    """
    logging.info("Cleaning data.")

    cleaned_dataset = []
    removed_item_counter = 0

    for datapoint in dataset:
        # np.isfinite returns true if number is not nan and not inf
        if np.isfinite(datapoint['distances']).all() and np.isfinite(datapoint['speed']).all():
            cleaned_dataset.append(datapoint)
        else:
            removed_item_counter += 1

    logging.info(f"Finished cleaning data, {removed_item_counter} items removed.")

    return cleaned_dataset, removed_item_counter


def merge_distance_speed(distance_values: dict,
                         speed_values: dict,
                         k: int,
                         hide_id_frame: bool = True
                         ) -> list[dict]:
    """
    Groups the input of the neural network (distance values) and the output
      of the neural network (speed value) together per (PedID, FrameID) Tuple.
      Only if a (PedID, FrameID) Tuple has both distance values and a speed value,
      it is included in the resulting dataset.
      The resulting dataset is flattened as well.

    :param distance_values: The input of the neural network, the distance values.
    :param speed_values: The output of the neural network, the speed values.
    :param k: The number of nearest neighbors.
    :param hide_id_frame: Whether to hide the PedID and FrameID in the result.
                          Only used for debugging, PedID and FrameID are not included by default.
    :return: A flattened list containing all complete datapoints in the dataset.
    """
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
    """
    Gets data, returns complete list of dicts.

    :param data:
    :param k:
    :param clean_data:
    :param hide_id_frame:
    :return:
    """

    # logging.basicConfig(level=logging.INFO)
    # logging.warning("Testwarning")

    # 1. get Input Values per ID+Frame
    logging.info("Preprocessing distance values.")
    trees_dict = construct_kNN_trees(data)
    distance_values = get_ped_distances(trees_dict, k=k)

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
