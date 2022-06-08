"""Created by Constantin Philippenko, 4th April 2022."""
import random
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.decomposition import PCA, IncrementalPCA
from torch.utils.data import DataLoader

from src.Client import Client, ClientsNetwork
from src.Constants import NB_CLIENTS, DEBUG, INPUT_TYPE, OUTPUT_TYPE, NB_LABELS, BATCH_SIZE
from src.FeaturesLearner import ReshapeTransform
from src.PytorchScaler import StandardScaler

FLAMBY_PATH = '../FLamby'

sys.path.insert(0, FLAMBY_PATH)
import flamby
sys.path.insert(0, FLAMBY_PATH + '/flamby')
import datasets

DIRICHLET_COEF = 0.5
PCA_NB_COMPONENTS = 16


def iid_split(data: np.ndarray, labels: np.ndarray, nb_clients: int,
              nb_points_by_non_iid_clients: np.array) -> [List[np.ndarray], List[np.ndarray]]:
    nb_points = data.shape[0]
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


def create_clients(nb_clients: int, data: np.ndarray, labels: np.ndarray, nb_labels: int, split: bool,
                   dataset_name: str, iid: bool = False) -> List[Client]:
    clients = []
    if split:
        # At this point, data are still splitted by clients following the natural order.
        nb_points_by_non_iid_clients = np.array([x.shape[0] for x in data])
    else:
        # TODO : in the case of dataset without natural split, the iid clients have balanced number of points.
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

    for i in range(nb_clients):
        clients.append(Client(i, X[i], Y[i], nb_labels, dataset_name))
    return clients


def features_representation(data, dataset_name):
    model = torch.load("src/saved_models/" + dataset_name + ".pt")
    return model(data)


def get_dataloader(fed_dataset, train, kwargs, kwargs_loader):
    dataset = fed_dataset(train=train, **kwargs)
    return DataLoader(dataset, batch_size=BATCH_SIZE, **kwargs_loader)


def compute_PCA_scikit(X: np.ndarray, PCA_size) -> np.ndarray:
    # n_components must be between 0 and min(n_samples, n_features).
    pca = PCA(n_components=PCA_size, svd_solver = 'randomized')
    pca.fit(X)
    return pca.transform(X)


def fit_PCA(loader: DataLoader, ipca_data: IncrementalPCA, ipca_labels: IncrementalPCA = None,
            scaler: StandardScaler = None) -> [IncrementalPCA, IncrementalPCA]:
    for x, y in loader:
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        if len(y.shape) > 2:
            y = torch.flatten(y, start_dim=1)
        if scaler is not None:
            x = scaler.transform(x)

        # To fit the PCA we must have a number of elements bigger than the PCA dimension, those we must drop the last
        # batch if it doesn't contain enough elements.
        if ipca_data is not None and x.shape[0] >= PCA_NB_COMPONENTS:
            ipca_data.partial_fit(x.numpy())
        if ipca_labels is not None and y.shape[0] >= PCA_NB_COMPONENTS:
            ipca_labels.partial_fit(y.numpy())
    return ipca_data, ipca_labels


def compute_PCA(loader: DataLoader, ipca_data: IncrementalPCA, ipca_labels: IncrementalPCA = None,
                scaler: StandardScaler = None) -> [np.ndarray, np.ndarray]:
    X, Y = [], []
    for x, y in loader:
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        if scaler is not None:
            x = scaler.transform(x)
        if ipca_data is not None:
            X.append(ipca_data.transform(x.numpy()))
        else:
            X.append(torch.flatten(x, start_dim=1).numpy())
        if ipca_labels is not None:
            Y.append(ipca_labels.transform(torch.flatten(y, start_dim=1).numpy()))
        else:
            Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


def normalize_data(loader: DataLoader, dataset_name: str, ipca_data: IncrementalPCA, ipca_labels: IncrementalPCA,
                   scaler: StandardScaler) -> [np.ndarray, np.ndarray]:

    if OUTPUT_TYPE[dataset_name] == "image":
        data, labels = compute_PCA(loader, ipca_data, ipca_labels, scaler)
    else:
        data, labels = compute_PCA(loader, ipca_data, ipca_labels=None, scaler=scaler)

    return data, labels


def get_processed_train_test_data(fed_dataset, dataset_name, ipca_data: IncrementalPCA,
                                  ipca_labels: IncrementalPCA, scaler: StandardScaler,
                                  kwargs, kwargs_loader={}) -> [np.ndarray, np.ndarray]:

    loader = get_dataloader(fed_dataset, train=True, kwargs=kwargs, kwargs_loader=kwargs_loader)
    data_train, labels_train = normalize_data(loader, dataset_name, ipca_data, ipca_labels, scaler)

    loader = get_dataloader(fed_dataset, train=False, kwargs=kwargs, kwargs_loader=kwargs_loader)
    data_test, labels_test = normalize_data(loader, dataset_name, ipca_data, ipca_labels, scaler)

    return np.concatenate([data_train, data_test]), np.concatenate([labels_train, labels_test])


def fit_PCA_train_test(fed_dataset, ipca_data, ipca_labels, scaler: StandardScaler, kwargs, kwargs_loader={}):

    loader_train = get_dataloader(fed_dataset, train=True, kwargs=kwargs, kwargs_loader=kwargs_loader)
    ipca_data, ipca_labels = fit_PCA(loader_train, ipca_data, ipca_labels, scaler)
    loader_test = get_dataloader(fed_dataset, train=False, kwargs=kwargs, kwargs_loader=kwargs_loader)
    ipca_data, ipca_labels = fit_PCA(loader_test, ipca_data, ipca_labels, scaler)

    print("Fit Incremental PCA - done.")
    return ipca_data, ipca_labels

def compute_mean_std(fed_dataset, kwargs, kwargs_loader={}):

    # Computing the mean
    loader = get_dataloader(fed_dataset, train=True, kwargs=kwargs, kwargs_loader=kwargs_loader)
    total_sum = 0
    nb_points = 0
    for x, y in loader:
        dims = list(range(x.dim() - 1))
        total_sum += torch.sum(x, dim=dims)
        nb_points += x.shape[0]
    loader = get_dataloader(fed_dataset, train=False, kwargs=kwargs, kwargs_loader=kwargs_loader)
    for x, y in loader:
        total_sum += torch.sum(x, dim=dims)
        nb_points += x.shape[0]
    mean = total_sum / nb_points

    # Computing the standard deviation
    loader = get_dataloader(fed_dataset, train=True, kwargs=kwargs, kwargs_loader=kwargs_loader)
    sum_of_squared_error = 0
    for x, y in loader:
        dims = list(range(x.dim() - 1))
        sum_of_squared_error += torch.sum((x - mean).pow(2), dim=dims)
    loader = get_dataloader(fed_dataset, train=False, kwargs=kwargs, kwargs_loader=kwargs_loader)
    for x, y in loader:
        sum_of_squared_error += torch.sum((x - mean).pow(2), dim=dims)
    std = torch.sqrt(sum_of_squared_error / nb_points)

    return mean, std


def fit_scaler(fed_dataset, dataset_name, kwargs):
    # We normalize only if the data are tabular
    if INPUT_TYPE[dataset_name] == "tabular":
        mean, std = compute_mean_std(fed_dataset, kwargs)
        return StandardScaler(mean=mean, std=std)
    return None


def get_dataset(dataset_name: str, features_learner: bool = False) -> [List[torch.FloatTensor], List[torch.FloatTensor]]:

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        # We reshape mnist to match with our neural network
        ReshapeTransform((-1,))

    ])

    ipca_data = IncrementalPCA(n_components=PCA_NB_COMPONENTS)
    ipca_labels = IncrementalPCA(n_components=PCA_NB_COMPONENTS) if OUTPUT_TYPE[dataset_name] == "image" else None

    if dataset_name == "mnist":
        from torchvision import datasets
        kwargs = dict(root='../DATASETS/MNIST', download=False, transform=transform)
        kwargs_loader = dict(shuffle=False)
        scaler = fit_scaler(datasets.MNIST, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(datasets.MNIST, ipca_data, ipca_labels, scaler, kwargs)
        X, Y = get_processed_train_test_data(datasets.MNIST, dataset_name, ipca_data, ipca_labels, scaler, kwargs,
                                             kwargs_loader)
        return X, Y, False

    elif dataset_name == "fashion_mnist":
        from torchvision import datasets
        from torchvision import datasets
        kwargs = dict(root='../DATASETS/FASHION_MNIST', download=False, transform=transform)
        kwargs_loader = dict(shuffle=False)
        scaler = fit_scaler(datasets.FashionMNIST, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(datasets.FashionMNIST, ipca_data, ipca_labels, scaler, kwargs)
        data, labels = get_processed_train_test_data(datasets.FashionMNIST,  dataset_name, ipca_data, ipca_labels,
                                                     scaler, kwargs, kwargs_loader)
        return data, labels, False

    elif dataset_name == "camelyon16":
        from datasets.fed_camelyon16.dataset import FedCamelyon16, collate_fn
        X, Y = [], []
        nb_of_client = 1 if DEBUG else NB_CLIENTS[dataset_name]
        ipca_data = ipca_data if not DEBUG else None
        kwargs_loader = dict(collate_fn=collate_fn)
        kwargs = dict(pooled=True, debug=DEBUG)
        scaler = fit_scaler(FedCamelyon16, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(FedCamelyon16, ipca_data, ipca_labels, scaler, kwargs, kwargs_loader)
        for i in range(nb_of_client):
            kwargs = dict(center=i, pooled=False, debug=DEBUG)
            data, labels = get_processed_train_test_data(FedCamelyon16, dataset_name, ipca_data, ipca_labels, scaler, kwargs,
                                                         kwargs_loader)
            X.append(data)
            Y.append(labels)
        # In debug mode, there is only 5 pictures from the first center.
        if DEBUG:
            X, Y = [X[0][:2], X[0][2:]], [Y[0][:2], Y[0][2:]]
        return X, Y, True

    elif dataset_name == "heart_disease":
        import datasets
        from datasets.fed_heart_disease.dataset import FedHeartDisease
        X, Y = [], []
        kwargs = dict(pooled=True)
        scaler = fit_scaler(FedHeartDisease, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(FedHeartDisease, ipca_data, ipca_labels, scaler, kwargs)
        for i in range(NB_CLIENTS[dataset_name]):
            kwargs = dict(center=i, pooled=False)
            data, labels = get_processed_train_test_data(FedHeartDisease, dataset_name, ipca_data, ipca_labels, scaler, kwargs)
            X.append(data)
            Y.append(labels)
        return X, Y, True

    elif dataset_name == "isic2019":
        from datasets.fed_isic2019.dataset import FedIsic2019
        X, Y = [], []
        kwargs = dict(pooled=True)
        scaler = fit_scaler(FedIsic2019, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(FedIsic2019, ipca_data, ipca_labels, scaler, kwargs)
        for i in range(NB_CLIENTS[dataset_name]):
            kwargs = dict(center=i, pooled=False)
            data, labels = get_processed_train_test_data(FedIsic2019, dataset_name, ipca_data, ipca_labels, scaler, kwargs)
            X.append(data)
            Y.append(labels)
        return X, Y, True

    elif dataset_name == "ixi":
        from datasets.fed_ixi.dataset import FedIXITiny
        X, Y = [], []
        kwargs = dict(pooled=True)
        scaler = fit_scaler(FedIXITiny, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(FedIXITiny, ipca_data, ipca_labels, scaler, kwargs)
        for i in range(NB_CLIENTS[dataset_name]):
            kwargs = dict(center=i, pooled=False)
            data, labels = get_processed_train_test_data(FedIXITiny, dataset_name, ipca_data, ipca_labels, scaler,
                                                         kwargs)
            X.append(data)
            Y.append(labels)
        return X, Y, True

    elif dataset_name == "kits19":
        from datasets.fed_kits19.dataset import FedKits19
        X, Y = [], []
        kwargs = dict(pooled=True)
        scaler = fit_scaler(FedKits19, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(FedKits19, ipca_data, ipca_labels, scaler, kwargs)
        for i in range(NB_CLIENTS[dataset_name]):
            kwargs = dict(center=i, pooled=False)
            data, labels = get_processed_train_test_data(FedKits19, dataset_name, ipca_data, ipca_labels, scaler, kwargs)
            X.append(data)
            Y.append(labels)
        return X, Y, True

    elif dataset_name == "lidc_idri":
        from datasets.fed_lidc_idri.dataset import FedLidcIdri
        X, Y = [], []
        kwargs = dict(pooled=True)
        scaler = fit_scaler(FedLidcIdri, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(FedLidcIdri, ipca_data, ipca_labels, scaler, kwargs)
        for i in range(NB_CLIENTS[dataset_name]):
            kwargs = dict(center=i, pooled=False)
            data, labels = get_processed_train_test_data(FedLidcIdri, dataset_name, ipca_data, ipca_labels, scaler, kwargs)
            X.append(data)
            Y.append(labels)
        return X, Y, True

    elif dataset_name == "tcga_brca":
        from datasets.fed_tcga_brca.dataset import FedTcgaBrca
        X, Y = [], []
        kwargs = dict(pooled=True)
        scaler = fit_scaler(FedTcgaBrca, dataset_name, kwargs)
        ipca_data, ipca_labels = fit_PCA_train_test(FedTcgaBrca, ipca_data, ipca_labels, scaler, kwargs)
        for i in range(NB_CLIENTS[dataset_name]):
            kwargs = dict(center=i, pooled=False)
            data, labels = get_processed_train_test_data(FedTcgaBrca, dataset_name, ipca_data, ipca_labels, scaler, kwargs)
            X.append(data)
            Y.append(labels[:,1].reshape(-1, 1))
        # plot_distrib(Y, 0, 1)
        # plot_distrib(Y, 0, 2)
        # plot_distrib(Y, 0, 3)
        # plot_distrib(Y, 4, 5)
        # plot_Y_histogram(Y)
        return X, Y, True

    raise ValueError("{0}: the dataset is unknown.".format(dataset_name))


def load_data(data: np.array, labels: np.array, splitted: bool, dataset_name: str, nb_clients: int,
              labels_type: str, iid: bool = False) -> ClientsNetwork:

    print("Regenerating clients.")
    print("Data shape:", data[0].shape[1:])
    print("Labels shape:", labels[0].shape[1:])
    start = time.time()

    nb_labels = NB_LABELS[dataset_name]

    clients = create_clients(nb_clients, data, labels, nb_labels, splitted, dataset_name, iid=iid)

    clients_network = ClientsNetwork(dataset_name, clients, None, labels_type, iid)

    print("Elapsed time: {:.2f}s".format(time.time() - start))

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

