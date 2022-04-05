"""Created by Constantin Philippenko, 4th April 2022."""
import ot
import torchvision.datasets as datasets

import numpy as np

from Client import Client, ClientsNetwork
from PickleHandler import pickle_loader
from StatisticalMetrics import StatisticalMetrics

DIRICHLET_COEF = 0.8
NB_CLIENTS = 10

DATASET_NAME = "mnist"

NB_LABEL = 10

mnist = datasets.MNIST(root='../DATASET', train=True, download=True, transform=None)
mnist_data = mnist.train_data.numpy()
mnist_data = mnist_data.reshape(mnist_data.shape[0], mnist_data.shape[1]*mnist_data.shape[2])
mnist_label = mnist.train_labels


def dirichlet_split(data, labels):
    X = [[] for i in range(NB_CLIENTS)]
    Y = [[] for i in range(NB_CLIENTS)]
    for idx_label in range(NB_LABEL):
        proportions = np.random.dirichlet(np.repeat(DIRICHLET_COEF, NB_CLIENTS))
        print("Proportions for label {0}: {1}".format(idx_label, proportions))
        assert round(proportions.sum()) == 1, "The sum of proportion is not equal to 1."
        last_idx = 0
        N = len(labels[labels == idx_label])
        for idx_client in range(NB_CLIENTS):
            X[idx_client].append(data[labels == idx_label][last_idx:last_idx + int(proportions[idx_client] * N)])
            Y[idx_client].append(labels[labels == idx_label][last_idx:last_idx + int(proportions[idx_client] * N)])
            last_idx += int(proportions[idx_client] * N)
    for idx_client in range(NB_CLIENTS):
        X[idx_client] = np.concatenate((X[idx_client]))
        Y[idx_client] = np.concatenate(Y[idx_client])
    return X, Y


def compute_KL_distance(distrib1, distrib2):
    EPS = 10e-6
    N = len(distrib1)
    return np.sum([distrib1[i] * np.log((distrib1[i] / (distrib2[i]+ EPS)) + EPS) for i in range(N)])


def compute_TV_distance(distrib1, distrib2):
    N = len(distrib1)
    return np.sum([np.abs(distrib1[i] - distrib2[i]) for i in range(N)])

def compute_EMD(distrib1, distrib2):
    cost_matrix = ot.dist(distrib1, distrib2)
    cost_matrix /= cost_matrix.max()
    a, b = ot.unif(len(distrib1)), ot.unif(len(distrib2))  # uniform distribution on samples
    return ot.bregman.sinkhorn2(a, b, cost_matrix, reg=0.1)  # Wasserstein distance / EMD value


def create_clients(nb_clients, data, labels):
    clients = []
    X, Y = dirichlet_split(data, labels)
    for i in range(nb_clients):
        clients.append(Client(i, X[i], Y[i], NB_LABEL))
    return clients


if __name__ == '__main__':

    my_metrics = StatisticalMetrics(DATASET_NAME, NB_CLIENTS)

    try:
        clients_network = pickle_loader("pickle/{0}/clients_network".format(DATASET_NAME))
        average_client = clients_network.average_client
        clients = clients_network.clients
    except:
        average_client = Client("central", mnist_data, mnist_label, NB_LABEL)
        clients = create_clients(NB_CLIENTS, mnist_data, mnist_label)
        clients_network = ClientsNetwork(DATASET_NAME, clients, average_client)

    # Vector of distance between client and the average.
    KL_distance_to_average = np.zeros(NB_CLIENTS)
    TV_distance_to_average = np.zeros(NB_CLIENTS)

    # Matrix of distance between each clients
    KL_joint_distance = np.zeros((NB_CLIENTS, NB_CLIENTS))
    TV_joint_distance = np.zeros((NB_CLIENTS, NB_CLIENTS))

    for i in range(NB_CLIENTS):
        KL_distance_to_average[i] = compute_KL_distance(clients[i].Y_distribution, average_client.Y_distribution)
        TV_distance_to_average[i] = compute_TV_distance(clients[i].Y_distribution, average_client.Y_distribution)

    # Compute TV distance (symmetric matrix)
    for i in range(NB_CLIENTS):
        for j in range(i, NB_CLIENTS):
            TV_joint_distance[i, j] = compute_TV_distance(clients[i].Y_distribution, clients[j].Y_distribution)
            TV_joint_distance[j, i] = TV_joint_distance[i, j]

    # Compute KL distance
    for i in range(NB_CLIENTS):
        for j in range(NB_CLIENTS):
            KL_joint_distance[i, j] = compute_KL_distance(clients[i].Y_distribution, clients[j].Y_distribution)

    my_metrics.set_metrics_on_Y(KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance)
    my_metrics.save_itself()

    print("KL distance to average:", KL_distance_to_average)
    print("TV distance to average:", TV_distance_to_average)

    # Vector of EMD between client and the average.
    EMD_to_average = np.zeros(NB_CLIENTS)

    # Matrix of EAD between each clients
    joint_EMD = np.zeros((NB_CLIENTS, NB_CLIENTS))

    # Compute Earth Mover's distance
    print("Computing EMD between clients")
    for i in range(NB_CLIENTS):
        for j in range(i, NB_CLIENTS):
            print(j)
            joint_EMD[i, j] = compute_EMD(clients[i].X_TSNE, clients[j].X_TSNE)
            joint_EMD[j, i] = joint_EMD[i, j]

    my_metrics.set_metrics_on_X(EMD_to_average, joint_EMD)
    my_metrics.save_itself()

    print("Computing EMD with average clients")
    for i in range(NB_CLIENTS):
        EMD_to_average[i] = compute_EMD(clients[i].X_TSNE, average_client.X_TSNE)

    my_metrics.plot_Y_metrics()
    my_metrics.plot_X_metrics()

