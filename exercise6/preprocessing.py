"""
Preprocessing Methods

TODO: Finalize Docstring(s)

"""

import numpy as np


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
