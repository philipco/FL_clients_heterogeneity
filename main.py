"""Created by Constantin Philippenko, 5th April 2022."""
import ot
import torchvision.datasets as datasets

import numpy as np
from tqdm import tqdm

from Client import Client, ClientsNetwork
from DataLoading import load_data
from PickleHandler import pickle_loader
from StatisticalMetrics import StatisticalMetrics

DIRICHLET_COEF = 0.8
NB_CLIENTS = 10

DATASET_NAME = "mnist"

NB_LABEL = 10


def compute_KL_distance(distrib1, distrib2):
    assert len(distrib1) == len(distrib2), "KL: distributions are not of equal size."
    EPS = 10e-6
    N = len(distrib1)
    return np.sum([distrib1[i] * np.log((distrib1[i] / (distrib2[i]+ EPS)) + EPS) for i in range(N)])


def compute_TV_distance(distrib1, distrib2):
    assert len(distrib1) == len(distrib2), "TV: distributions are not of equal size."
    N = len(distrib1)
    return np.sum([np.abs(distrib1[i] - distrib2[i]) for i in range(N)])

def compute_EMD(distrib1, distrib2):
    cost_matrix = ot.dist(distrib1, distrib2)
    cost_matrix /= cost_matrix.max()
    a, b = ot.unif(len(distrib1)), ot.unif(len(distrib2))  # uniform distribution on samples
    return ot.bregman.sinkhorn2(a, b, cost_matrix, reg=0.1)  # Wasserstein distance / EMD value


if __name__ == '__main__':

    my_metrics = StatisticalMetrics(DATASET_NAME, NB_CLIENTS)

    clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=True)
    clients = clients_network.clients
    average_client = clients_network.average_client

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
    my_metrics.plot_Y_metrics()

    print("KL distance to average:", KL_distance_to_average)
    print("TV distance to average:", TV_distance_to_average)

    # Vector of EMD between client and the average.
    EMD_to_average = np.zeros(NB_CLIENTS)

    # Matrix of EAD between each clients
    joint_EMD = np.zeros((NB_CLIENTS, NB_CLIENTS))

    # Compute Earth Mover's distance
    print("Computing EMD with average clients ...")
    for i in tqdm(range(NB_CLIENTS)):
        EMD_to_average[i] = compute_EMD(clients[i].X_TSNE, average_client.X_TSNE)

    print("Computing EMD between clients ...")
    for i in tqdm(range(NB_CLIENTS)):
        print("Client ", i)
        for j in range(i, NB_CLIENTS):
            joint_EMD[i, j] = compute_EMD(clients[i].X_TSNE, clients[j].X_TSNE)
            joint_EMD[j, i] = joint_EMD[i, j]

    my_metrics.set_metrics_on_X(EMD_to_average, joint_EMD)
    my_metrics.save_itself()

    my_metrics.plot_X_metrics()

