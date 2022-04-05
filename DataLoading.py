"""Created by Constantin Philippenko, 4th April 2022."""
import numpy as np
from torchvision import datasets

from Client import Client, ClientsNetwork
from PickleHandler import pickle_loader

DIRICHLET_COEF = 0.7


def dirichlet_split(data, labels, nb_clients):
    nb_labels = len(np.unique(labels)) # Here data is not yet split. Thus nb_labels is correct.
    X = [[] for i in range(nb_clients)]
    Y = [[] for i in range(nb_clients)]
    for idx_label in range(nb_labels):
        proportions = np.random.dirichlet(np.repeat(DIRICHLET_COEF, nb_clients))
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


def create_clients(nb_clients, data, labels):
    clients = []
    nb_labels = len(np.unique(labels)) # Here data is not yet split. Thus nb_labels is correct.
    X, Y = dirichlet_split(data, labels, nb_clients)
    for i in range(nb_clients):
        clients.append(Client(i, X[i], Y[i], nb_labels))
    return clients


def load_data(dataset_name, nb_clients, recompute: bool = False):
    if not recompute:
        clients_network = pickle_loader("pickle/{0}/clients_network".format(dataset_name))
        # average_client = clients_network.average_client
        # clients = clients_network.clients
    else:
        mnist = datasets.MNIST(root='../DATASET', train=True, download=True, transform=None)
        mnist_data = mnist.train_data.numpy()
        mnist_data = mnist_data.reshape(mnist_data.shape[0], mnist_data.shape[1] * mnist_data.shape[2])
        mnist_label = mnist.train_labels

        nb_labels = len(np.unique(mnist_label))

        average_client = Client("central", mnist_data, mnist_label, nb_labels)
        clients = create_clients(nb_clients, mnist_data, mnist_label)
        clients_network = ClientsNetwork(dataset_name, clients, average_client, nb_labels)

    return clients_network

