"""Created by Constantin Philippenko, 29th September 2022."""
import warnings
from random import random
from typing import List

import numpy as np
from sklearn.manifold import TSNE


def split_data(features: List[np.array], labels: List[np.array], natural_split: bool, nb_of_clients: int):
    if natural_split:
        nb_points_by_clients = [len(l) for l in labels]
        features_iid, labels_iid = iid_split(np.concatenate(features), np.concatenate(labels), nb_of_clients,
                                             nb_points_by_clients)
        features_heter, labels_heter = features, labels
    else:
        features, labels = np.concatenate(features), np.concatenate(labels)
        nb_points_by_clients = np.array([len(features) // nb_of_clients for i in range(nb_of_clients)])
        features_iid, labels_iid = iid_split(features, labels, nb_of_clients, nb_points_by_clients)
        features_heter, labels_heter = iid_split(features, labels, nb_of_clients, nb_points_by_clients)
    return features_iid, labels_iid, features_heter, labels_heter, nb_points_by_clients


def iid_split(data: np.ndarray, labels: np.ndarray, nb_clients: int,
              nb_points_by_non_iid_clients: np.array) -> [List[np.ndarray], List[np.ndarray]]:
    nb_points = data.shape[0]
    X = []
    Y = []
    indices = np.arange(nb_points)
    np.random.shuffle(indices)
    idx_split = [np.sum(nb_points_by_non_iid_clients[:i]) for i in range(1, nb_clients)]
    split_indices = np.array_split(indices, idx_split)
    for i in range(nb_clients):
        X.append(data[split_indices[i]])
        Y.append(labels[split_indices[i]])
    return X, Y


def dirichlet_split(data: np.ndarray, labels: np.ndarray, nb_clients: int, dirichlet_coef: float) \
        -> [List[np.ndarray], List[np.ndarray]]:
    nb_labels = len(np.unique(labels)) # Here data is not yet split. Thus nb_labels is correct.
    X = [[] for i in range(nb_clients)]
    Y = [[] for i in range(nb_clients)]
    for idx_label in range(nb_labels):
        proportions = np.random.dirichlet(np.repeat(dirichlet_coef, nb_clients))
        assert round(proportions.sum()) == 1, "The sum of proportion is not equal to 1."
        last_idx = 0
        N = len(labels[labels == idx_label])
        for idx_client in range(nb_clients):
            X[idx_client].append(data[labels == idx_label][last_idx:last_idx + int(proportions[idx_client] * N)])
            Y[idx_client].append(labels[labels == idx_label][last_idx:last_idx + int(proportions[idx_client] * N)])
            last_idx += int(proportions[idx_client] * N)

            # If the client hasn't receive this kind of label we add at least one !
            if len(X[idx_client][-1]) == 0:
                random_idx = random.randint(0,len(data[labels == idx_label]) - 1)
                X[idx_client][-1] = data[labels == idx_label][random_idx:random_idx+1]
                Y[idx_client][-1] = labels[labels == idx_label][random_idx:random_idx+1]

    for idx_client in range(nb_clients):
        X[idx_client] = np.concatenate((X[idx_client]))
        Y[idx_client] = np.concatenate(Y[idx_client])
    return X, Y


def compute_TSNE(self, X: np.ndarray):
    """Compute the TSNE representation of a dataset."""
    print("Computing TSNE of client {0}".format(self.idx))
    np.random.seed(25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne = TSNE(n_components=2, random_state=42)
        embedded_data = tsne.fit_transform(X)
    return embedded_data