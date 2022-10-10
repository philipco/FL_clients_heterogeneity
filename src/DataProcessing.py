"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from src.Constants import PCA_NB_COMPONENTS
from src.Data import DataDecentralized, DataCentralized
from src.PickleHandler import pickle_saver
from src.Split import iid_split
from src.Utilities import create_folder_if_not_existing
from src.UtilitiesNumpy import fit_PCA, compute_PCA


def decentralized_processing_of_data(dataset_name: str, features: List[np.array], labels: List[np.array],
                                     batch_size: int, nb_of_clients: int) -> DataDecentralized:
    nb_points_by_clients = [len(l) for l in labels]
    features_iid, labels_iid = iid_split(np.concatenate(features), np.concatenate(labels), nb_of_clients,
                                         nb_points_by_clients)

    data = DataDecentralized(dataset_name, nb_points_by_clients, features_iid, features,
                             labels_iid, labels, batch_size)

    create_folder_if_not_existing("pickle/{0}/processed_data".format(dataset_name))
    pickle_saver(data, "pickle/{0}/processed_data/decentralized".format(dataset_name))

    return data


def centralized_processing_of_data(dataset_name: str, features: List[np.array], labels: List[np.array],
                                   batch_size: int, nb_of_clients: int) -> DataCentralized:

    # Scaling and computing a PCA for the complete dataset once for all.
    scaler = StandardScaler().fit(np.concatenate(features))
    ipca_data = IncrementalPCA(n_components=PCA_NB_COMPONENTS)
    ipca_data = fit_PCA(np.concatenate(features), ipca_data, scaler, batch_size)
    features = [compute_PCA(f, ipca_data, scaler, batch_size) for f in features]

    nb_points_by_clients = [len(l) for l in labels]
    features_iid, labels_iid = iid_split(np.concatenate(features), np.concatenate(labels), nb_of_clients,
                                         nb_points_by_clients)

    data = DataCentralized(dataset_name, nb_points_by_clients, features_iid, features, labels_iid, labels)

    create_folder_if_not_existing("pickle/{0}/processed_data".format(dataset_name))
    pickle_saver(data, "pickle/{0}/processed_data/centralized".format(dataset_name))

    return data