"""Created by Constantin Philippenko, 4th April 2022."""
import random
import sys
from typing import List

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torchvision
from torch.utils.data import DataLoader

from src.Client import Client, ClientsNetwork
from src.Constants import NB_CLIENTS, DEBUG
from src.FeaturesLearner import ReshapeTransform

DIRICHLET_COEF = 0.5
PCA_NB_COMPONENTS = 10


def iid_split(data: np.ndarray, labels: np.ndarray, nb_clients: int, nb_points_by_non_iid_clients: np.array)\
        -> [List[np.ndarray], List[np.ndarray]]:
    nb_points = len(labels)
    X = []
    Y = []
    indices = np.arange(nb_points)
    np.random.shuffle(indices)
    idx_split = [np.sum(nb_points_by_non_iid_clients[:i]) for i in range(1, nb_clients)]
    split_indices = np.array_split(indices, idx_split)
    for i in range(nb_clients):
        X.append(data[split_indices[i]])
        Y.append(labels[split_indices[i]])
    return X, Y


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


def create_clients(nb_clients: int, data: np.ndarray, labels: np.ndarray, nb_labels: int, split: bool, labels_type: str,
                   iid: bool = False) -> List[Client]:
    clients = []
    if split:
        nb_points_by_non_iid_clients = np.array([len(y) for y in labels])
    else:
        nb_points_by_non_iid_clients = np.array([len(labels) // nb_clients for i in range(nb_clients)])
    # It the dataset is already split and we don't want to create an iid dataset.
    if split and not iid:
        X, Y = data, labels
    else:
        if split:
            data, labels = np.concatenate(data), np.concatenate(labels)
        if iid:
            X, Y = iid_split(data, labels, nb_clients, nb_points_by_non_iid_clients)
        else:
            X, Y = dirichlet_split(data, labels, nb_clients, dirichlet_coef=DIRICHLET_COEF)
    # TODO
    # assert [len(np.unique(y)) for y in Y] == [nb_labels for y in Y], "Some labels are not represented on some clients."
    PCA_size = min(PCA_NB_COMPONENTS, min([len(x) for x in X]))
    for i in range(nb_clients):
        clients.append(Client(i, X[i], Y[i], nb_labels, PCA_size, labels_type))
    return clients, PCA_size


def features_representation(data, dataset_name):
    model = torch.load("src/saved_models/" + dataset_name + ".pt")
    return model(data)


def get_dataset(dataset_name: str, features_learner: bool = True) -> [np.ndarray, np.ndarray]:

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        # We reshape mnist to match with our neural network
        ReshapeTransform((-1,))

    ])

    if dataset_name == "mnist":
        from torchvision import datasets
        mnist_dataset = datasets.MNIST(root='../DATASETS/MNIST', train=True, download=True, transform=transform)
        dataloader = DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)
        mnist_data, mnist_labels = next(iter(dataloader))
        if features_learner:
            mnist_data = features_representation(mnist_data, dataset_name).detach().numpy()
        mnist_labels = mnist_labels.numpy()
        return mnist_data, mnist_labels, False

    elif dataset_name == "fashion_mnist":
        from torchvision import datasets
        mnist_dataset = datasets.FashionMNIST(root='../../DATASETS/FASHION_MNIST', train=True, download=True,
                                              transform=None)
        dataloader = DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)
        mnist_data, mnist_labels = next(iter(dataloader))
        if features_learner:
            mnist_data = features_representation(mnist_data, dataset_name).detach().numpy()
        mnist_labels = mnist_labels.numpy()
        return mnist_data, mnist_labels, False

    elif dataset_name == "camelyon16":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        from datasets.fed_camelyon16.dataset import FedCamelyon16, collate_fn
        X, Y = [], []
        nb_of_client = 1 if DEBUG else NB_CLIENTS[dataset_name]
        for i in range(nb_of_client):
            train_dataset = FedCamelyon16(train=True, pooled=False, center=i, debug=DEBUG)
            training_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), collate_fn=collate_fn)
            for s, (data, labels) in enumerate(training_dataloader):
                if features_learner:
                    data = features_representation(data, dataset_name).detach()
                X.append(data.reshape(data.shape[0], data.shape[1] * data.shape[2]).numpy())
                Y.append(np.concatenate(labels.numpy()))
        # In debug mode, there is only 5 pictures from the first center.
        if DEBUG:
            X, Y = [X[0][:2], X[0][2:]], [Y[0][:2], Y[0][2:]]
        return X, Y, True

    elif dataset_name == "isic2019":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        from datasets.fed_isic2019.dataset import FedIsic2019
        X, Y = [], []
        nb_of_client = 1 if DEBUG else NB_CLIENTS[dataset_name]
        sz = 200
        train_aug = albumentations.Compose(
            [
                albumentations.RandomCrop(sz, sz),
                albumentations.Normalize(always_apply=True),
            ]
        )
        for i in range(NB_CLIENTS[dataset_name]):
            print(i)
            train_dataset = FedIsic2019(train=True, pooled=False, center=i, augmentations=train_aug)
            data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
            if features_learner:
                data = features_representation(data, dataset_name).detach()
            X.append(data.reshape(data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]).numpy())
            Y.append(labels.numpy())
        return X, Y, True

    elif dataset_name == "ixi":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        from datasets.fed_ixi.dataset import FedIXITiny
        X, Y = [], []
        for i in range(NB_CLIENTS[dataset_name]):
            train_dataset = FedIXITiny(train=True, pooled=False, center=i)
            data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
            data = data.reshape(data.shape[0], data.shape[1] * data.shape[2] * data.shape[3] * data.shape[4])
            if features_learner:
                data = features_representation(data, dataset_name).detach()
            X.append(data.numpy())
            Y.append(np.concatenate(labels.numpy()))
        return X, Y, True

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
        X, Y = [], []
        for i in range(NB_CLIENTS[dataset_name]):
            train_dataset = FedTcgaBrca(train=True, pooled=False, center=i)
            data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
            if features_learner:
                data = features_representation(data, dataset_name).detach()
            X.append(data.numpy())
            Y.append(labels.numpy()[:,1].reshape(-1, 1))
        # plot_distrib(Y, 0, 1)
        # plot_distrib(Y, 0, 2)
        # plot_distrib(Y, 0, 3)
        # plot_distrib(Y, 4, 5)
        # plot_Y_histogram(Y)
        return X, Y, True

    elif dataset_name == "heart_disease":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        from datasets.fed_heart_disease.dataset import FedHeartDisease
        X, Y = [], []
        for i in range(NB_CLIENTS[dataset_name]):
            train_dataset = FedHeartDisease(train=True, pooled=False, center=i)
            data, labels = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
            if features_learner:
                data = features_representation(data, dataset_name).detach()
            X.append(data.numpy())
            Y.append(np.concatenate(labels.numpy()))
        return X, Y, True

    raise ValueError("{0}: the dataset is unknown.".format(dataset_name))


def load_data(data: np.array, labels: np.array, splitted: bool, dataset_name: str, nb_clients: int,
              labels_type: str, iid: bool = False) -> ClientsNetwork:

    print("Regenerating clients.")

    if splitted:
        nb_labels = len(np.unique(np.concatenate(labels)))
    else:
        nb_labels = len(np.unique(labels))

    clients, PCA_size = create_clients(nb_clients, data, labels, nb_labels, splitted, labels_type, iid=iid)
    if splitted:
        central_client = Client("central", np.concatenate(data), np.concatenate(labels), nb_labels, PCA_size,
                                labels_type)
    else:
        central_client = Client("central", data, labels, nb_labels, PCA_size, labels_type)

    clients_network = ClientsNetwork(dataset_name, clients, central_client, labels_type, iid)

    return clients_network


def plot_distrib(Y, idx1, idx2):
    # fig, axes = plt.subplots(2, 1)

    plt.plot(np.sort(np.concatenate(Y[idx1])), label=int(idx1))
    plt.plot(np.sort(np.concatenate(Y[idx2])), label=int(idx2))
    plt.xlabel("Nb of points")
    plt.ylabel("Values")
    plt.title("Client {0} vs client {1}".format(idx1, idx2))
    plt.show()

    # axes[0].plot(distrib1)
    # axes[1].plot(distrib2)
    # plt.show()

def plot_cost_matrix(distrib1, distrib2):
    a, b = np.sort(np.concatenate(distrib1)).reshape(-1, 1), np.sort(np.concatenate(distrib2)).reshape(-1, 1)
    # use fast 1D solver
    import ot
    cost_matrix = ot.dist(a, b)

    # Equivalent to
    # G0 = ot.emd(a, b, M)
    import matplotlib.pylab as pl
    import ot.plot
    pl.figure(figsize=(5, 5))
    ot.plot.plot1D_mat(a, b, cost_matrix, 'OT matrix G0')
    a, b = ot.unif(len(distrib1)), ot.unif(len(distrib2))  # uniform distribution on samples
    print(ot.emd2(a, b, cost_matrix))
    pl.show()

def plot_Y_histogram(distrib):
    for i in range(len(distrib)):
        d = distrib[i]
        plt.hist(d, bins=10, label="client {0}".format(i), alpha=0.5)
    plt.legend()
    plt.show()

