"""
Module implementing the PedestrianDataset class.
Also implementing the function create_dataset() and other helpful functions
used for creating a PedestrianDataset from a .txt file.

author: Simon Blöchinger
"""

import logging
import os
import typing as t
from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import preprocessing


class PedestrianDataType(Enum):
    """
    The PedestrianDataType Enum is used to only allow certain values in the data loading functions.
    Available values are 'bottleneck', 'corridor' and 'all'.
    """
    BOTTLENECK = 'bottleneck'
    CORRIDOR = 'corridor'
    ALL = 'all'


class PedestrianDataset(Dataset):
    """
    The PedestrianDataset class, implementing a basic dataset.
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def _split_preprocessed_data(preprocessed_data: list[dict],
                             train_test_split: tuple[int, int] = (.5, .5),
                             cross_val_subset_count: int = 5
                             ) -> tuple[list[list[dict]], list[dict]]:
    """
    Splits already preprocessed data into multiple train/val datasets and one test dataset.

    :param preprocessed_data: The data that has already been preprocessed and will now be split into
        multiple train/val datasets and one test dataset.
    :param train_test_split: The percentage of the data used for the train/val dataset and the test dataset.
    :param cross_val_subset_count: The amount of train/val dataset parts (used for cross validation).
    :return: A list of `cross_val_subset_count` train/val dataset parts,
        that should be used to train the network with cross validation.
        Also returns a test dataset for testing the network.
    """
    assert np.sum(train_test_split) == 1

    # Choose a fixed random seed, so results are reproducible
    randomness = torch.Generator().manual_seed(1234)

    # 1. Split into Train+Val and Test

    # Calculate random indices
    indices = torch.randperm(len(preprocessed_data), generator=randomness)

    # Divide the random indices into two subsets for the train+val set and the test set
    indices_train_val = indices[:int(train_test_split[0] * len(preprocessed_data))]
    indices_test = indices[int(train_test_split[0] * len(preprocessed_data)):]

    # Append the test data directly to the final list
    test_data = [preprocessed_data[i] for i in indices_test]

    # 2. Split Train+Val into multiple Subsets for cross-validation
    train_val_data_list = []
    for i in range(cross_val_subset_count):
        indices_train_val_part = indices_train_val[
                                    int(i * len(indices_train_val) / cross_val_subset_count):
                                    int((i + 1) * len(indices_train_val) / cross_val_subset_count)
                                 ]
        train_val_data_list.append([preprocessed_data[i] for i in indices_train_val_part])

    return train_val_data_list, test_data


def _create_datasets_single_file(file_path: str,
                                 k: int = 10,
                                 train_test_split: tuple[int, int] = (.5, .5),
                                 cross_val_subset_count: int = 5,
                                 ) -> tuple[list[Dataset], Dataset]:
    """
    Creates multiple PedestrianDatasets from a single file.
    The data from that file is preprocessed and split into a list of training/validation datasets that are intended
    to be used with cross validation, and one test dataset which is intended for testing the neural network.

    :param file_path: The path to the file that should be loaded.
    :param k: The number of considered neighbors for the kNN algorithm.
    :param train_test_split: The percentage of the data used for the train/val dataset and the test dataset.
    :param cross_val_subset_count: The amount of train/val dataset parts (used for cross validation).
    :return: A list of `cross_val_subset_count` train/val dataset parts,
        that should be used to train the network with cross validation.
        Also returns a test dataset for testing the network.
    """
    logging.info(f"Creating dataset {file_path}")
    # Load data
    loaded_data = np.loadtxt(file_path)

    # Preprocess data
    preprocessed_data = preprocessing.do_preprocessing(loaded_data, k)

    # Split the data
    train_val_datas, test_data = _split_preprocessed_data(preprocessed_data, train_test_split, cross_val_subset_count)

    # Create datasets
    train_val_datasets = [PedestrianDataset(tvd) for tvd in train_val_datas]
    test_dataset = PedestrianDataset(test_data)

    return train_val_datasets, test_dataset


def _get_dataset_file_names(pedestrian_data_type: PedestrianDataType) -> list[str]:
    """
    Generates data file names from a PedestrianDataType for convenient loading of multiple files
    from one category.
    Can either return all filenames that contain bottleneck data, all filenames that contain
    corridor data, or all filenames in total.

    :param pedestrian_data_type: The PedestrianDataType value for which the list of files should be loaded.
    :return: A list of file names for the given PedestrianDataType.

    :raises:
        ValueError: If the PedestrianDataType is an unexpected value.
    """
    # The directory where the data is stored
    data_dir = "./Data/"

    # The values to generate the filenames for the bottleneck data with
    bottleneck_dir = "Bottleneck_Data/"
    bottleneck_widths = ['070', '095', '120', '180']
    bottleneck_paths = [os.path.join(data_dir,
                                     bottleneck_dir,
                                     (lambda w: f"uo-180-{w}.txt")(w)) for w in bottleneck_widths
                        ]

    # The values to generate the filenames for the corridor data with
    corridor_dir = "Corridor_Data/"
    corridor_peoplecounts = ['015', '030', '060', '085', '095', '110', '140', '230']
    corridor_paths = [os.path.join(data_dir,
                                   corridor_dir,
                                   (lambda p: f"ug-180-{p}.txt")(p)) for p in corridor_peoplecounts
                      ]

    # Return the
    if pedestrian_data_type == PedestrianDataType.BOTTLENECK:
        return bottleneck_paths
    elif pedestrian_data_type == PedestrianDataType.CORRIDOR:
        return corridor_paths
    elif pedestrian_data_type == PedestrianDataType.ALL:
        return bottleneck_paths + corridor_paths
    else:
        raise ValueError("Invalid PedestrianDataType")


def create_dataset(data_identifier: t.Union[PedestrianDataType, list[str]],
                   k: int = 10,
                   train_test_split: tuple[int, int] = (.5, .5),
                   cross_val_subset_count: int = 5,
                   ) -> (list[Dataset], Dataset):
    """
    Creates multiple PedestrianDatasets from files.
    Can load either a list of files or all files from one of the categories 'bottleneck', 'corridor' and 'all'.
    The data is preprocessed and split into a list of training/validation datasets that are intended to be
    used with cross validation, and one test dataset which is intended for testing the neural network.
    When splitting the data into the different datasets, it is ensured that an equal portion of data from
    each file is put into each of the datasets, to avoid some of the datasets containing vastly different
    data from other datasets.

    :param data_identifier: The files that should be loaded. A `PedestrianDataType` can be given instead of a list of files.
    :param k: The number of considered neighbors for the kNN algorithm.
    :param train_test_split: The percentage of the data used for the train/val dataset and the test dataset.
    :param cross_val_subset_count: The amount of train/val dataset parts (used for cross validation).
    :return: A list of `cross_val_subset_count` train/val dataset parts,
        that should be used to train the network with cross validation.
        Also returns a test dataset for testing the network.
    """
    logging.info(f"Creating Dataset for {data_identifier} with k={k}, "
                 f"train_test_split={train_test_split}, cross_val_subset_count={cross_val_subset_count}.")

    # If data_identifier is a PedestrianDataType, load the file names corresponding to that PedestrianDataType
    if isinstance(data_identifier, PedestrianDataType):
        data_file_names = _get_dataset_file_names(data_identifier)
    else:
        data_file_names = data_identifier

    # Load datasets for each file and store them in lists
    train_val_datasets_list = []
    test_dataset_list = []
    for data_file_name in data_file_names:
        train_val_d, test_d = _create_datasets_single_file(data_file_name, k, train_test_split, cross_val_subset_count)
        train_val_datasets_list.append(train_val_d)
        test_dataset_list.append(test_d)

    # Concatenate the datasets from the lists

    # # Concat train_val datasets (used with cross validation)
    train_val_datasets = []
    transposed_train_val_datasets_list = list(map(list, zip(*train_val_datasets_list)))
    for sublist in transposed_train_val_datasets_list:
        train_val_datasets.append(ConcatDataset(sublist))

    # # Concat test datasets
    test_dataset = ConcatDataset(test_dataset_list)

    return train_val_datasets, test_dataset
