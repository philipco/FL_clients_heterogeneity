"""Created by Constantin Philippenko, 4th April 2022."""
import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale, StandardScaler
import seaborn as sns

from PickleHandler import pickle_saver, pickle_loader
from Utilities import create_folder_if_not_existing

NB_CLUSTER_ON_X = 5
NB_COMPONENTS = 5


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

        # Now that all clients, and also the average client are ready, we can compute the Y|X distribution.
        # To compute Y given X distribution, we need to first compute X cluster on complete distribution.
        self.compute_Y_given_X_distribution()
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
        
    def clusterize_X_distribution_for_each_client(self):
        X_cluster_predictor = self.train_X_cluster_predictor()
        for client in self.clients:
                client.compute_X_clusters(X_cluster_predictor)
        
    def train_X_cluster_predictor(self):
        X_cluster_predictor = KMeans(n_clusters=NB_CLUSTER_ON_X, random_state=0).fit(self.average_client.X_lower_dim)
        self.average_client.set_X_clusters(X_cluster_predictor.labels_)
        return X_cluster_predictor

    def compute_Y_given_X_distribution(self):
        self.clusterize_X_distribution_for_each_client()
        self.average_client.set_Y_given_X_distribution()
        for client in self.clients:
            client.set_Y_given_X_distribution()


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
        self.X = X
        # self.X_TSNE = self.compute_TSNE(self.X)
        self.X_PCA = self.compute_PCA(self.X)
        self.X_lower_dim = self.X_PCA
        self.Y = Y
        self.nb_labels = nb_labels

        # Distributions that we need to compare between clients.
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

    def compute_PCA(self, X):
        pca = make_pipeline(StandardScaler(), PCA(n_components=5))
        pca.fit(X)
        return pca.transform(X)

    def compute_Y_distribution(self):
        return np.array([(self.Y == y).sum() / len(self.Y) for y in range(self.nb_labels)])

    def compute_X_given_Y_distribution(self):
        distrib = [self.X_lower_dim[self.Y == y] for y in range(self.nb_labels)]
        assert [len(x) > 0 for x in distrib] == [True for x in distrib], "X|Y, some labels are missing."
        return distrib

    def set_Y_given_X(self):
        self.Y_given_X = [self.Y[self.X_clusters == x] for x in range(NB_CLUSTER_ON_X)]

    def set_Y_given_X_distribution(self):
        self.Y_given_X_distribution = []
        for x in range(NB_CLUSTER_ON_X):
            self.Y_given_X_distribution.append(np.array([(self.Y[self.X_clusters == x] == y).sum() / len(self.Y[self.X_clusters == x]) for y in range(self.nb_labels)]))

    def set_X_clusters(self, X_clusters):
        self.X_clusters = X_clusters
    
    def compute_X_clusters(self, X_cluster_predictor):
        self.set_X_clusters(X_cluster_predictor.predict(self.X_lower_dim))



