"""Created by Constantin Philippenko, 4th April 2022."""
import warnings

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

from PickleHandler import pickle_saver
from Utilities import create_folder_if_not_existing


class ClientsNetwork:

    def __init__(self, dataset_name, clients, average_client) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.clients = clients
        self.average_client = average_client
        self.nb_clients = len(clients)
        self.save_itself()

    def save_itself(self):
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/clients_network".format(self.dataset_name))


class Client:

    def __init__(self, idx, X, Y, nb_labels) -> None:
        super().__init__()
        self.idx = idx
        self.X = X
        self.X_TSNE = self.compute_TSNE(X)
        self.Y = Y
        self.nb_labels = nb_labels
        self.Y_distribution = self.compute_Y_distribution()
        self.X_given_Y_distribution = self.compute_X_given_Y_distribution()

    def compute_TSNE(self, X):
        """Compute the TSNE representation of a dataset."""
        print("Computing TSNE of client {0}".format(self.idx))
        np.random.seed(25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne = TSNE()
            embedded_data = tsne.fit_transform(scale(X))
        return embedded_data

    def compute_Y_distribution(self):
        return np.array([len(self.X[self.Y == y]) / len(self.Y) for y in range(self.nb_labels)])

    def compute_X_given_Y_distribution(self):
        pass
