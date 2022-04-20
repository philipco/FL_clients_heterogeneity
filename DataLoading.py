"""Created by Constantin Philippenko, 4th April 2022."""

import numpy as np
from torchvision import datasets

from Client import Client, ClientsNetwork
from PickleHandler import pickle_loader

DIRICHLET_COEF = 0.5


def dirichlet_split(data, labels, nb_clients, dirichlet_coef):
    nb_labels = len(np.unique(labels)) # Here data is not yet split. Thus nb_labels is correct.
    X = [[] for i in range(nb_clients)]
    Y = [[] for i in range(nb_clients)]
    for idx_label in range(nb_labels):
        proportions = np.random.dirichlet(np.repeat(dirichlet_coef, nb_clients))
        print("Proportions for label {0}: {1}".format(idx_label, proportions))
        assert round(proportions.sum()) == 1, "The sum of proportion is not equal to 1."
        last_idx = 0
        N = len(labels[labels == idx_label])
        for idx_client in range(nb_clients):
            X[idx_client].append(data[labels == idx_label][last_idx:last_idx + int(proportions[idx_client] * N)])
            Y[idx_client].append(labels[labels == idx_label][last_idx:last_idx + int(proportions[idx_client] * N)])
            last_idx += int(proportions[idx_client] * N)
    for idx_client in range(nb_clients):
        X[idx_client] = np.concatenate((X[idx_client]))
        Y[idx_client] = np.concatenate(Y[idx_client])
    return X, Y


def create_clients(nb_clients, data, labels, iid: bool = False):
    clients = []
    nb_labels = len(np.unique(labels)) # Here data is not yet split. Thus nb_labels is correct.
    if iid:
        X, Y = dirichlet_split(data, labels, nb_clients, dirichlet_coef=10**5)
    else:
        X, Y = dirichlet_split(data, labels, nb_clients, dirichlet_coef=DIRICHLET_COEF)
    for i in range(nb_clients):
        clients.append(Client(i, X[i], Y[i], nb_labels))
    return clients


def load_data(dataset_name, nb_clients, recompute: bool = False, iid: bool = False):
    if iid:
        dataset_name = "{0}-iid".format(dataset_name)

    if not recompute:
        clients_network = pickle_loader("pickle/{0}/clients_network".format(dataset_name))
        for client in clients_network.clients:
            client.Y_distribution = client.compute_Y_distribution()
    else:
        mnist = datasets.MNIST(root='../DATASETS', train=True, download=True, transform=None)
        mnist_data = mnist.train_data.numpy()
        mnist_data = mnist_data.reshape(mnist_data.shape[0], mnist_data.shape[1] * mnist_data.shape[2])
        mnist_label = mnist.train_labels

        print("LEN ======= ", len(mnist_data))
        mnist_data = mnist_data[:50000]
        mnist_label = mnist_label[:50000]

        nb_labels = len(np.unique(mnist_label))

        average_client = Client("central", mnist_data, mnist_label, nb_labels)
        clients = create_clients(nb_clients, mnist_data, mnist_label, iid=iid)

        clients_network = ClientsNetwork(dataset_name, clients, average_client, nb_labels)

    return clients_network

