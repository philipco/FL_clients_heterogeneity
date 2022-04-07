"""Created by Constantin Philippenko, 5th April 2022."""
import math

import ot
import torchvision.datasets as datasets

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from Client import Client, ClientsNetwork
from DataLoading import load_data
from PickleHandler import pickle_loader
from StatisticalMetrics import StatisticalMetrics

DIRICHLET_COEF = 0.8
NB_CLIENTS = 10

NB_CLUSTER_ON_X = 5

DATASET_NAME = "mnist"

NB_LABELS = 10


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
    try:
        cost_matrix /= cost_matrix.max()
    except:
        return math.nan # When distribution1 is empty.
    a, b = ot.unif(len(distrib1)), ot.unif(len(distrib2))  # uniform distribution on samples
    return ot.bregman.sinkhorn2(a, b, cost_matrix, reg=0.1)  # Wasserstein distance / EMD value


def compute_metrics_on_Y(clients, average_client):

    # Vector of distance between client and the average.
    KL_distance_to_average = np.zeros(NB_CLIENTS)
    TV_distance_to_average = np.zeros(NB_CLIENTS)

    # Matrix of distance between each clients
    KL_distance_one_to_one = np.zeros((NB_CLIENTS, NB_CLIENTS))
    TV_distance_one_to_one = np.zeros((NB_CLIENTS, NB_CLIENTS))

    for i in range(NB_CLIENTS):
        KL_distance_to_average[i] = compute_KL_distance(clients[i].Y_distribution, average_client.Y_distribution)
        TV_distance_to_average[i] = compute_TV_distance(clients[i].Y_distribution, average_client.Y_distribution)

    # Compute TV distance (symmetric matrix)
    for i in range(NB_CLIENTS):
        for j in range(i, NB_CLIENTS):
            TV_distance_one_to_one[i, j] = compute_TV_distance(clients[i].Y_distribution, clients[j].Y_distribution)
            TV_distance_one_to_one[j, i] = TV_distance_one_to_one[i, j]

    # Compute KL distance
    for i in range(NB_CLIENTS):
        for j in range(NB_CLIENTS):
            KL_distance_one_to_one[i, j] = compute_KL_distance(clients[i].Y_distribution, clients[j].Y_distribution)
            
    return KL_distance_to_average, TV_distance_to_average, KL_distance_one_to_one, TV_distance_one_to_one


def compute_metrics_on_X(clients, average_client: Client):
    print("\n=== Compute metrics on X ===")

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
        for j in range(i, NB_CLIENTS):
            joint_EMD[i, j] = compute_EMD(clients[i].X_TSNE, clients[j].X_TSNE)
            joint_EMD[j, i] = joint_EMD[i, j]

    return EMD_to_average, joint_EMD


def compute_metrics_on_X_given_Y(clients, average_client):

    # Vector of EMD between client and the average.
    EMD_to_average = [np.zeros(NB_CLIENTS) for y in range(NB_LABELS)]

    # Matrix of EAD between each clients
    joint_EMD = [np.zeros((NB_CLIENTS, NB_CLIENTS)) for y in range(NB_LABELS)]

    print("\n=== Compute metrics on X given Y ===")
    for y in range(NB_LABELS):

        # Compute Earth Mover's distance
        print("Computing EMD with average clients ...")
        for i in tqdm(range(NB_CLIENTS)):
            EMD_to_average[y][i] = compute_EMD(clients[i].X_given_Y_distribution[y], average_client.X_given_Y_distribution[y])

        print("Computing EMD between clients ...")
        for i in tqdm(range(NB_CLIENTS)):
            for j in range(i, NB_CLIENTS):
                joint_EMD[y][i, j] = compute_EMD(clients[i].X_given_Y_distribution[y], clients[j].X_given_Y_distribution[y])
                joint_EMD[y][j, i] = joint_EMD[y][i, j]

    return EMD_to_average, joint_EMD


def compute_metrics_on_Y_given_X(clients, average_client):

    # Vector of distance between client and the average.
    KL_distance_to_average = [np.zeros(NB_CLIENTS) for x in range(NB_CLUSTER_ON_X)]
    TV_distance_to_average = [np.zeros(NB_CLIENTS) for x in range(NB_CLUSTER_ON_X)]

    # Matrix of distance between each clients
    KL_joint_distance = [np.zeros((NB_CLIENTS, NB_CLIENTS)) for x in range(NB_CLUSTER_ON_X)]
    TV_joint_distance = [np.zeros((NB_CLIENTS, NB_CLIENTS)) for x in range(NB_CLUSTER_ON_X)]

    print("\n=== Compute metrics on X given Y ===")
    for x in tqdm(range(NB_CLUSTER_ON_X)):

        for i in range(NB_CLIENTS):
            KL_distance_to_average[x][i] = compute_KL_distance(clients[i].Y_given_X_distribution[x],
                                                               average_client.Y_given_X_distribution[x])
            TV_distance_to_average[x][i] = compute_TV_distance(clients[i].Y_given_X_distribution[x],
                                                               average_client.Y_given_X_distribution[x])

        # Compute TV distance (symmetric matrix)
        for i in range(NB_CLIENTS):
            for j in range(i, NB_CLIENTS):
                TV_joint_distance[x][i, j] = compute_TV_distance(clients[i].Y_given_X_distribution[x],
                                                                 clients[j].Y_given_X_distribution[x])
                TV_joint_distance[x][j, i] = TV_joint_distance[x][i, j]

        # Compute KL distance
        for i in range(NB_CLIENTS):
            for j in range(NB_CLIENTS):
                KL_joint_distance[x][i, j] = compute_KL_distance(clients[i].Y_given_X_distribution[x],
                                                                 clients[j].Y_given_X_distribution[x])

    return KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance

if __name__ == '__main__':

    my_metrics = StatisticalMetrics(DATASET_NAME, NB_CLIENTS, NB_LABELS)
    clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=False)
    clients = clients_network.clients
    average_client = clients_network.average_client

    # KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance = compute_metrics_on_Y(clients, average_client)
    # my_metrics.set_metrics_on_Y(KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance)
    # my_metrics.save_itself()
    # my_metrics.plot_Y_metrics([my_metrics.TV_distance_one_to_one, my_metrics.KL_distance_one_to_one,
    #                            my_metrics.TV_distance_to_average, my_metrics.KL_distance_to_average], "Y")
    #
    EMD_to_average, joint_EMD = compute_metrics_on_X(clients, average_client)
    my_metrics.set_metrics_on_X(EMD_to_average, joint_EMD)
    my_metrics.save_itself()
    my_metrics.plot_X_metrics([my_metrics.EMD_one_to_one, my_metrics.EMD_to_average], "X")
    #
    # EMD_by_Y_to_average, EMD_by_Y_one_to_one = compute_metrics_on_X_given_Y(clients, average_client)
    # my_metrics.set_metrics_on_X_given_Y(EMD_by_Y_to_average, EMD_by_Y_one_to_one)
    # my_metrics.save_itself()
    # my_metrics.plot_X_given_Y_metrics()

    # KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance = compute_metrics_on_Y_given_X(clients, average_client)
    # my_metrics.set_metrics_on_Y_given_X(KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance)
    # my_metrics.save_itself()
    # my_metrics.plot_Y_given_X_metrics()

