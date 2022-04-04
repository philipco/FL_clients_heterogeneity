"""Created by Constantin Philippenko, 4th April 2022."""
import torch
import torchvision
import torchvision.datasets as datasets

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Client import Client

DIRICHLET_COEF = 0.8
NB_CLIENTS = 10

NB_LABEL = 10

mnist = datasets.MNIST(root='../DATASET', train=True, download=True, transform=None)
mnist_data = mnist.train_data.numpy()
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
    return np.sum([distrib1[i] * np.log(distrib1[i] / (distrib2[i] + EPS)) for i in range(N)])


def compute_TV_distance(distrib1, distrib2):
    N = len(distrib1)
    return np.sum([np.abs(distrib1[i] - distrib2[i]) for i in range(N)])


def create_clients(nb_clients, data, labels):
    clients = []
    X, Y = dirichlet_split(data, labels)
    for i in range(nb_clients):
        clients.append(Client(X[i], Y[i], NB_LABEL))
    return clients

if __name__ == '__main__':

    average_client = Client(mnist_data, mnist_label, NB_LABEL)
    clients = create_clients(NB_CLIENTS, mnist_data, mnist_label)

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

    # Compute KL distance (symmetric matrix)
    for i in range(NB_CLIENTS):
        for j in range(NB_CLIENTS):
            KL_joint_distance[i, j] = compute_KL_distance(clients[i].Y_distribution, clients[j].Y_distribution)

    print("KL distance to average:", KL_distance_to_average)
    print("TV distance to average:", TV_distance_to_average)

    # fig, axes = plt.subplots(2, 2, figsize=(12, 3))

    fig = plt.figure()

    ax1 = plt.subplot2grid((NB_CLIENTS+1, 2), (0, 0), colspan=1, rowspan=NB_CLIENTS)
    ax2 = plt.subplot2grid((NB_CLIENTS+1, 2), (0, 1), colspan=1, rowspan=NB_CLIENTS)
    ax3 = plt.subplot2grid((NB_CLIENTS+1, 2), (NB_CLIENTS, 0), colspan=1, rowspan=1)
    ax4 = plt.subplot2grid((NB_CLIENTS+1, 2), (NB_CLIENTS, 1), colspan=1, rowspan=1)

    titles = ["TV distance between clients", "KL distance between clients"]
    matrix_to_plot = [TV_joint_distance, KL_joint_distance, TV_distance_to_average, KL_distance_to_average]
    axes = [ax1, ax2, ax3, ax4]

    for i in range(len(matrix_to_plot)):
        if len(matrix_to_plot[i].shape) != 1:
            im = axes[i].imshow(matrix_to_plot[i], cmap="Blues")
            axes[i].set_title(label=titles[i])
        else:
            im = axes[i].imshow(np.expand_dims(matrix_to_plot[i], axis=0), cmap="Blues")
            plt.axis('off')
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if i < 2:
            plt.colorbar(im, ax=axes[i], cax=cax)
    plt.show()
