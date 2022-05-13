"""Created by Constantin Philippenko, 4th April 2022."""
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
import seaborn as sns
from typing import List

from src.PickleHandler import pickle_saver, pickle_loader
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

    def __init__(self, idx: int, X: np.ndarray, Y: np.ndarray, nb_labels: int, PCA_size: int, labels_type: str) -> None:
        super().__init__()
        self.idx = idx
        self.X = X
        # self.X_TSNE = self.compute_TSNE(self.X)

        # If there is less than 20 elements, or less that 20 dimensions.
        # There is less than 20 elements for Camelyon in debug mode.
        if self.X.shape[1] <= 20 or self.X.shape[0] <= 20:
            scaler = preprocessing.StandardScaler().fit(self.X)
            self.X_lower_dim = scaler.transform(self.X)
        else:
            self.PCA_size = PCA_size
            self.X_PCA = self.compute_PCA(self.X)
            self.X_lower_dim = self.compute_PCA(self.X)
        self.Y = Y
        self.nb_labels = nb_labels

        # Distributions that we need to compare between clients.
        self.Y_distribution = self.compute_Y_distribution(labels_type)

    def compute_TSNE(self, X: np.ndarray):
        """Compute the TSNE representation of a dataset."""
        print("Computing TSNE of client {0}".format(self.idx))
        np.random.seed(25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tsne = TSNE(n_components=2, random_state=42)
            embedded_data = tsne.fit_transform(X)
        return embedded_data

    def compute_PCA(self, X: np.ndarray) -> np.ndarray:
        # n_components must be between 0 and min(n_samples, n_features).
        pca = make_pipeline(StandardScaler(), PCA(n_components=self.PCA_size))
        pca.fit(X)
        return pca.transform(X)

    def compute_Y_distribution(self, labels_type: str) -> np.ndarray:
        if labels_type == "discrete":
            return np.array([(self.Y == y).sum() / len(self.Y) for y in range(self.nb_labels)])
        elif labels_type == "continuous":
            return self.Y
        else:
            raise ValueError("Unrecognized labels type.")


    def compute_X_given_Y_distribution(self, labels_type: str) -> np.ndarray:
        if labels_type == "discrete":
            distrib = [self.X_lower_dim[self.Y == y] for y in range(self.nb_labels)]
            assert [len(x) > 0 for x in distrib] == [True for x in distrib], "X|Y, some labels are missing."
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
                    np.array([(self.Y[self.X_clusters == x] == y).sum() / len(self.Y[self.X_clusters == x])
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

    def __init__(self, dataset_name: str, clients: List[Client], centralized_client: Client, labels_type: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.clients = clients
        self.centralized = centralized_client
        self.nb_clients = len(clients)
        self.min_len_dataset = min([len(client.Y) for client in self.clients])

        # Now that all clients, and also the centralized client are ready, we can compute the Y|X distribution.
        # To compute Y given X distribution, we need to first compute X cluster on complete distribution.
        self.labels_type = labels_type
        self.compute_Y_given_X_distribution()
        self.compute_X_given_Y_distribution()

        self.save_itself()

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