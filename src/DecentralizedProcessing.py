"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import numpy as np
import torch
from sklearn.decomposition import PCA

from src.Constants import NB_CLIENTS
from src.Data import Data, DataDecentralized
from src.PickleHandler import pickle_saver
from src.Split import iid_split, split_data
from src.Utilities import create_folder_if_not_existing

def get_processed_data(dataset_name: str, features: List[np.array], labels: List[np.array], natural_split: bool) \
        -> DataDecentralized:

    # nb_of_points = np.sum([len(f) for f in features])
    nb_of_clients = NB_CLIENTS[dataset_name]

    features_iid, labels_iid, features_heter, labels_heter, nb_points_by_clients = \
        split_data(features, labels, natural_split, nb_of_clients)

    data = DataDecentralized(dataset_name, nb_points_by_clients, natural_split, features_iid, features_heter,
                             labels_iid, labels_heter)

    create_folder_if_not_existing("pickle/{0}/processed_data".format(dataset_name))
    pickle_saver(data, "pickle/{0}/processed_data/decentralized".format(dataset_name))

    return data