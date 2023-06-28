"""
Preprocessing Methods

TODO: Finalize Docstring(s)

"""

import numpy as np
import scipy


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


def get_ped_speeds(ped_paths):
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


def get_kNN(ped_data):
    # Compute kNN Tree for each Frame sequentially
    ped_frames = get_ped_frames(ped_data)
    kNN_frames = {}

    for key, frame in ped_frames.items():
        kNN_frames[key] = scipy.spatial.KDTree(frame[:, 2:5])

    return kNN_frames


def build_data_structure(trees_dict, k=10):
    ks = [k for k in range(2, k+2)]  # build the k's without [1] so that self isn't included
    data_structure_dict = {}

    for key, tree in trees_dict.items():
        item_dict = {}  # Holds input and Truth Data

        # 1. Calculate the Input Data of the following Shape:
        # Length: 2K+1
        # mean_spacing - x_diff - y_diff - x_diff2 - y_diff 2 - ...
        # Note: Why split in x and y? Why not just distance?
        nn_array = np.ones((tree.n, 2*len(ks) + 1))

        for i in range(tree.n):
            # returns distances, indexes
            distances, indexes = tree.query(tree.data[i], k=ks)

            # calculate mean spacing
            nn_array[i, 0] = np.mean(distances)

            # calculate y_diff and x_diff
            for j, k in enumerate(indexes):
                if k < tree.n:
                    nn_array[i, 2*j+1] = tree.data[k, 0] - tree.data[i, 0]  # x_diff
                    nn_array[i, 2*j+2] = tree.data[k, 1] - tree.data[i, 1]  # y_diff

                else:
                    print(f"Tree: {key}\tTreeIndex: {i}\tNeighborIndex: {k}\n"
                          f"\tk: {len(ks)}\tTreeItems: {tree.n}\tConsideredTreeItems: {tree.n - 1}\n"  # -1 Item considered since self is removed
                          f"\tNot enough neighbors, adding np.inf to result array.")
                    # If not enough neighbors are there, np.inf will be added to the array.
                    # This is also the default behaviour in the returned distances
                    # Maybe this doesn't make sense for training the NN though, not sure what we could do when we do not
                    #   have enough neighbors there. Will test with larger dataset soon.
                    nn_array[i, 2 * j + 1] = np.inf
                    nn_array[i, 2 * j + 2] = np.inf

        item_dict['x'] = nn_array

        # 2. Calculate the Truth Data (the actual speed of the pedestrian) that the network will predict

        # TODO Calculate truth data
        # Difficulty: Need to consider Frame Axis with PedID constant
        # Here: Frame constant
        # -> Maybe calculate speed separately? And then add it later according to frameID and PedID?

        # TODO extra note: This probably needs to be 1 more layer inside

        item_dict['y'] = None  # TODO

        data_structure_dict[key] = nn_array  # TODO: change that when complete
        # data_structure_dict[key] = item_dict

    return data_structure_dict





