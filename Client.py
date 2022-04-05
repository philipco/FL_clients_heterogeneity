"""Created by Constantin Philippenko, 4th April 2022."""
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
import seaborn as sns

from PickleHandler import pickle_saver, pickle_loader
from Utilities import create_folder_if_not_existing

NB_POINTS_TSNE = 1000


def palette(nb_of_labels, labels):
    complete_palette = sns.color_palette("bright", nb_of_labels)
    return [complete_palette[i] for i in labels]


class ClientsNetwork:

    def __init__(self, dataset_name, clients, average_client, nb_labels: int) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.clients = clients
        self.average_client = average_client
        self.nb_clients = len(clients)
        # self.nb_labels = nb_labels
        self.save_itself()

    def save_itself(self):
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/clients_network".format(self.dataset_name))

    def plot_TSNE(self):
        tsne_folder = "pictures/{0}/tsne".format(self.dataset_name)
        create_folder_if_not_existing(tsne_folder)
        for i in range(self.nb_clients):
            scatter_plot(self.clients[i].X_TSNE, self.clients[i].Y, self.clients[i].idx, tsne_folder)
        scatter_plot(self.average_client.X_TSNE, self.average_client.Y, self.average_client.idx, tsne_folder)


def scatter_plot(data, labels, idx, tsne_folder):
    fig, ax = plt.subplots(figsize=(12, 9))
    my_palette = palette(10, np.unique(labels))
    sns.scatterplot(data[:, 0], data[:, 1], ax=ax, hue=labels, legend='full', palette=my_palette) \
        .set_title("TSNE - 2D representation of client {0}".format(idx))
    plt.savefig('{0}/{1}.eps'.format(tsne_folder, idx), format='eps')


class Client:

    def __init__(self, idx, X, Y, nb_labels) -> None:
        super().__init__()
        self.idx = idx
        self.X = X[:NB_POINTS_TSNE]
        self.X_TSNE = self.compute_TSNE(self.X)
        self.Y = Y[:NB_POINTS_TSNE]
        self.nb_labels = nb_labels
        self.Y_distribution = self.compute_Y_distribution()
        self.X_given_Y_distribution = self.compute_X_given_Y_distribution()

    def compute_TSNE(self, X):
        """Compute the TSNE representation of a dataset."""
        print("Computing TSNE of client {0}".format(self.idx))
        np.random.seed(25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne = TSNE(n_components=2, random_state=42)
            embedded_data = tsne.fit_transform(X)
        return embedded_data

    def compute_Y_distribution(self):
        return np.array([len(self.X[self.Y == y]) / len(self.Y) for y in range(self.nb_labels)])

    def compute_X_given_Y_distribution(self):
        return [self.X_TSNE[self.Y == y] for y in range(self.nb_labels)]


