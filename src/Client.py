"""Created by Constantin Philippenko, 4th April 2022."""
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
import seaborn as sns
from typing import List

from statsmodels.distributions import ECDF

from src.Constants import INPUT_TYPE, OUTPUT_TYPE
from src.PickleHandler import pickle_saver
from src.PytorchScaler import StandardScaler
from src.Utilities import create_folder_if_not_existing

NB_CLUSTER_ON_CONTINUOUS_VAR = 3


def palette(nb_of_labels: int, labels: np.ndarray):
    complete_palette = sns.color_palette("bright", nb_of_labels)
    return [complete_palette[i] for i in labels]


def scatter_plot(data: np.ndarray, labels: np.ndarray, idx: int, tsne_folder: str):
    fig, ax = plt.subplots(figsize=(12, 9))
    my_palette = palette(10, np.unique(labels))
    sns.scatterplot(data[:, 0], data[:, 1], ax=ax, hue=labels, legend='full', palette=my_palette) \
        .set_title("TSNE - 2D representation of client {0}".format(idx))
    plt.savefig('{0}/{1}.eps'.format(tsne_folder, idx), format='eps')


class Client:

    def __init__(self, idx: int, X: torch.FloatTensor, Y: torch.FloatTensor, nb_labels: int, dataset_name: str) -> None:
        super().__init__()
        self.idx = idx
        self.nb_labels = nb_labels
        self.dataset_name = dataset_name

        self.X = X
        self.X_lower_dim = self.X
        self.Y = Y
        self.Y_distribution = self.compute_Y_distribution(OUTPUT_TYPE[dataset_name])



    def compute_Y_distribution(self, labels_type: str) -> torch.FloatTensor:
        if labels_type == "discrete":
            return torch.FloatTensor([(self.Y == y).sum() / len(self.Y) for y in range(self.nb_labels)])
        elif labels_type in ["continuous", "image"]:
            return self.Y
        else:
            raise ValueError("Unrecognized labels type.")

    def compute_X_given_Y_distribution(self, labels_type: str) -> np.ndarray:
        if labels_type == "discrete":
            distrib = [self.X_lower_dim[self.Y == y] for y in range(self.nb_labels)]
            # assert [len(x) > 0 for x in distrib] == [True for x in distrib], "X|Y, some labels are missing."
        elif labels_type == "continuous":
            distrib = [self.X_lower_dim[self.Y_clusters == y] for y in range(NB_CLUSTER_ON_CONTINUOUS_VAR)]
            # Je ne peux pas garantir que touts les clusters de Y seront bien sur chaque machine ...
        else:
            raise ValueError("Unrecognized labels type.")
        return distrib

    ### For Y | X ###
    def set_X_clusters(self, X_clusters) -> None:
        self.X_clusters = X_clusters

    def compute_X_clusters(self, X_cluster_predictor: KMeans) -> None:
        self.set_X_clusters(X_cluster_predictor.predict(self.X_lower_dim))

    def set_Y_given_X_distribution(self, labels_type: str) -> None:
        self.Y_given_X_distribution = []
        if labels_type == "discrete":
            for x in range(NB_CLUSTER_ON_CONTINUOUS_VAR):
                self.Y_given_X_distribution.append(
                    torch.FloatTensor([(self.Y[self.X_clusters == x] == y).sum() / len(self.Y[self.X_clusters == x])
                              for y in range(self.nb_labels)]))
        elif labels_type == "continuous":
            for x in range(NB_CLUSTER_ON_CONTINUOUS_VAR):
                self.Y_given_X_distribution.append(self.Y[self.X_clusters == x])
        else:
            raise ValueError("Unrecognized labels type.")

    ################
    ### For X | Y (relevant only when Y is continuous ###
    def set_Y_clusters(self, Y_clusters) -> None:
        self.Y_clusters = Y_clusters

    def compute_Y_clusters(self, Y_cluster_predictor: KMeans) -> None:
        self.set_Y_clusters(Y_cluster_predictor.predict(self.Y.reshape(-1, 1)))
    #####################################################


class ClientsNetwork:

    def __init__(self, dataset_name: str, clients: List[Client], centralized_client: Client, labels_type: str,
                 iid: bool) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.clients = clients
        self.centralized = centralized_client
        self.nb_clients = len(clients)
        self.min_len_dataset = min([len(client.Y) for client in self.clients])
        self.iid = iid
        self.metrics_folder = "pictures/" + self.dataset_name

        # Now that all clients, and also the centralized client are ready, we can compute the Y|X distribution.
        # To compute Y given X distribution, we need to first compute X cluster on complete distribution.
        self.labels_type = labels_type
        # self.compute_Y_given_X_distribution()
        # self.compute_X_given_Y_distribution()

        self.save_itself()

    def print_Y_empirical_distribution_function(self):
        fig, ax1 = plt.subplots(figsize=(12,8))
        ax2 = ax1.twinx()
        bins = np.histogram(np.hstack((np.concatenate(self.clients[idx].Y) for idx in range(len(self.clients)))),
                            bins=40)[1]
        for idx in range(len(self.clients)):
            # fit a cdf
            distrib_Y = np.concatenate(self.clients[idx].Y)
            ecdf = ECDF(distrib_Y)
            ax1.plot(ecdf.x, ecdf.y, label=idx)
            ax2.hist(distrib_Y, bins=bins, density=False, histtype='step', stacked=True, fill=False, lw=2, label=idx)
        ax1.set_xlabel("Range of labels' values", fontsize=15)
        ax2.set_ylabel("Frequency of labels' values", fontsize=15)
        ax1.set_ylabel("Cumulative probability", fontsize=15)
        iid_str = "iid" if self.iid else "non-iid"
        plt.title("Empirical distribution function (left) and histogram (right) for {0} ({1})".format(
            self.dataset_name, iid_str), fontsize=15
        )
        plt.legend()
        plt.savefig('{0}/{1}.png'.format(self.metrics_folder, "edf"), dvi=600, bbox_inches='tight')

    def save_itself(self) -> None:
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/clients_network".format(self.dataset_name))

    def plot_TSNE(self) -> None:
        tsne_folder = "pictures/{0}/tsne".format(self.dataset_name)
        create_folder_if_not_existing(tsne_folder)
        for i in range(self.nb_clients):
            scatter_plot(self.clients[i].X_TSNE, self.clients[i].Y, self.clients[i].idx, tsne_folder)
        scatter_plot(self.centralized.X_TSNE, self.centralized.Y, self.centralized.idx, tsne_folder)

    def compute_Y_given_X_distribution(self) -> None:
        # Computing clusters on X
        cluster_predictor = train_cluster_predictor(self.centralized.X_lower_dim)
        self.centralized.set_X_clusters(cluster_predictor.labels_)
        for client in self.clients:
            client.compute_X_clusters(cluster_predictor)

        # Computing Y|X.
        self.centralized.set_Y_given_X_distribution(self.labels_type)
        for client in self.clients:
            client.set_Y_given_X_distribution(self.labels_type)

    def compute_X_given_Y_distribution(self) -> None:

        if self.labels_type == "continuous":
            # Computing clusters on Y
            cluster_predictor = train_cluster_predictor(self.centralized.Y.reshape(-1, 1))
            self.centralized.set_Y_clusters(cluster_predictor.labels_)
            for client in self.clients:
                client.compute_Y_clusters(cluster_predictor)

        self.centralized.X_given_Y_distribution = self.centralized.compute_X_given_Y_distribution(self.labels_type)
        for client in self.clients:
            client.X_given_Y_distribution = client.compute_X_given_Y_distribution(self.labels_type)


def train_cluster_predictor(data) -> object:
    cluster_predictor = KMeans(n_clusters=NB_CLUSTER_ON_CONTINUOUS_VAR, random_state=0).fit(data)
    return cluster_predictor