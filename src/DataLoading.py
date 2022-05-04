"""Created by Constantin Philippenko, 4th April 2022."""
import random
import sys
from os import path
from typing import List

import numpy as np
from torch.utils.data import DataLoader

from src.Client import Client, ClientsNetwork
from src.PickleHandler import pickle_loader

DIRICHLET_COEF = 0.5
PCA_NB_COMPONENTS = 5


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


def create_clients(nb_clients: int, data: np.ndarray, labels: np.ndarray, iid: bool = False) -> List[Client]:
    clients = []
    nb_labels = len(np.unique(labels)) # Here data is not yet split. Thus nb_labels is correct.
    if iid:
        X, Y = dirichlet_split(data, labels, nb_clients, dirichlet_coef=10**5)
    else:
        X, Y = dirichlet_split(data, labels, nb_clients, dirichlet_coef=DIRICHLET_COEF)
    assert [len(np.unique(y)) for y in Y] == [len(np.unique(labels)) for y in Y], "Some labels are not represented on some clients."
    PCA_size = min(PCA_NB_COMPONENTS, min([len(x) for x in X]))
    for i in range(nb_clients):
        clients.append(Client(i, X[i], Y[i], nb_labels, PCA_size))
    return clients, PCA_size


def get_dataset(dataset_name: str) -> [np.ndarray, np.ndarray]:

    if dataset_name == "mnist":
        from torchvision import datasets
        mnist = datasets.MNIST(root='../../DATASETS', train=True, download=True, transform=None)
        mnist_data = mnist.train_data.numpy()
        mnist_data = mnist_data.reshape(mnist_data.shape[0], mnist_data.shape[1] * mnist_data.shape[2])
        mnist_label = mnist.train_labels.numpy()
        return mnist_data, mnist_label

    elif dataset_name == "fashion_mnist":
        from torchvision import datasets
        mnist = datasets.FashionMNIST(root='../../DATASETS', train=True, download=True, transform=None)
        mnist_data = mnist.train_data.numpy()
        mnist_data = mnist_data.reshape(mnist_data.shape[0], mnist_data.shape[1] * mnist_data.shape[2])
        mnist_label = mnist.train_labels.numpy()
        return mnist_data, mnist_label

    elif dataset_name == "isic2019":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        from datasets.fed_isic2019.dataset import FedIsic2019
        train_dataset = FedIsic2019(train=True, pooled=True)
        data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
        return data.numpy(), labels.numpy()

    elif dataset_name == "tcga_brca":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby/datasets/fed_tcga_brca')
        import dataset
        # Required to import correctly lifelines
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby/datasets/')
        import lifelines
        import fed_tcga_brca
        from datasets.fed_tcga_brca.dataset import FedTcgaBrca
        train_dataset = FedTcgaBrca(train=True, pooled=True) # TODO : not pooled
        data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
        return data.numpy(), labels[:,0].numpy() # TODO : weird labels size

    elif dataset_name == "heart_disease":
        pass

    raise ValueError("{0}: the dataset is unknown.".format(dataset_name))


def load_data(dataset_name: str, nb_clients: int, recompute: bool = False, iid: bool = False) -> ClientsNetwork:

    if not recompute:
        clients_network = pickle_loader("pickle/{0}/clients_network".format(dataset_name))
        for client in clients_network.clients:
            client.Y_distribution = client.compute_Y_distribution()
    else:
        print("Regenerating clients.")

        data, labels = get_dataset(dataset_name)

        nb_labels = len(np.unique(labels))

        clients, PCA_size = create_clients(nb_clients, data, labels, iid=iid)
        central_client = Client("central", data, labels, nb_labels, PCA_size)

        clients_network = ClientsNetwork(dataset_name, clients, central_client)

    return clients_network

