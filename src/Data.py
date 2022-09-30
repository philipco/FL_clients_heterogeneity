"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List, Union, Any

import numpy as np
from sklearn.decomposition import PCA

from src.Split import split_data


class Data:

    def __init__(self, dataset_name: str, nb_points_by_clients: List[int], natural_split: bool,
                 features_iid: List[np.array], features_heter: List[np.array], labels_iid: List[np.array],
                 labels_heter: List[np.array]) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.nb_points_by_clients = nb_points_by_clients
        self.nb_of_clients = len(features_heter)
        self.natural_split = natural_split

        self.features_iid = features_iid
        self.features_heter = features_heter
        self.labels_iid = labels_iid
        self.labels_heter = labels_heter

        self.labels_iid_distrib = compute_Y_distribution(self.labels_iid)
        self.labels_heter_distrib = compute_Y_distribution(self.labels_heter)

    def resplit_iid(self) -> None:
        features_iid, labels_iid, features_heter, labels_heter, nb_points_by_clients = \
            split_data(self.features_heter, self.labels_heter, self.natural_split, self.nb_of_clients)
        self.features_iid = features_iid
        self.features_heter = features_heter
        self.labels_iid = labels_iid
        self.labels_heter = labels_heter

        self.labels_iid_distrib = compute_Y_distribution(self.labels_iid)
        self.labels_heter_distrib = compute_Y_distribution(self.labels_heter)


class DataDecentralized(Data):

    def __init__(self, dataset_name: str, nb_points_by_clients: List[int], natural_split: bool,
                 features_iid: List[np.array], features_heter: List[np.array],
                 labels_iid: List[np.array], labels_heter: List[np.array]) -> None:
        super().__init__(dataset_name, nb_points_by_clients, natural_split, features_iid, features_heter, labels_iid,
                         labels_heter)
        print("Fitting decentralized PCA on iid split.")
        self.PCA_fit_iid = [PCA(n_components=10).fit(X) for X in self.features_iid]
        print("Fitting decentralized PCA on heterogeneous split.")
        self.PCA_fit_heter = [PCA(n_components=10).fit(X) for X in self.features_heter]

    def resplit_iid(self) -> None:
        super().resplit_iid()
        self.PCA_fit_iid = [PCA(n_components=10).fit(X) for X in self.features_iid]
        self.PCA_fit_heter = [PCA(n_components=10).fit(X) for X in self.features_heter]


def compute_Y_distribution(labels: List[np.array]) -> np.array:
    nb_labels = len(np.unique(np.concatenate(labels)))
    return [np.array([(l == y).sum() / len(l) for y in range(nb_labels)]) for l in labels]