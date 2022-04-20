"""Created by Constantin Philippenko, 20th April 2022."""
import math

import numpy as np
import ot
from tqdm import tqdm

from Client import Client

NB_CLUSTER_ON_X = 5


def compute_KL_distance(distrib1, distrib2):
    """Kullback-Leibler."""
    assert len(distrib1) == len(distrib2), "KL: distributions are not of equal size."
    EPS = 10e-6
    N = len(distrib1)
    return np.sum([distrib1[i] * np.log((distrib1[i] / (distrib2[i]+ EPS)) + EPS) for i in range(N)])


def compute_TV_distance(distrib1, distrib2):
    """Total Variation."""
    assert len(distrib1) == len(distrib2), "TV: distributions are not of equal size."
    N = len(distrib1)
    return 0.5 * np.sum([np.abs(distrib1[i] - distrib2[i]) for i in range(N)])


def compute_EM_distance(distrib1, distrib2):
    """Earth's mover."""
    cost_matrix = ot.dist(distrib1, distrib2)
    try:
        cost_matrix /= cost_matrix.max()
    except:
        return math.nan # When distribution1 is empty.
    a, b = ot.unif(len(distrib1)), ot.unif(len(distrib2))  # uniform distribution on samples
    return ot.bregman.sinkhorn2(a, b, cost_matrix, reg=0.1)  # Wasserstein distance / EMD value


def compute_metrics_on_Y(clients, average_client):
    print("\n=== Compute metrics on Y ===")
    nb_clients = len(clients)

    # Vector of distance between client and the average.
    KL_distance_to_average = np.zeros(nb_clients)
    TV_distance_to_average = np.zeros(nb_clients)

    # Matrix of distance between each clients
    KL_distance_one_to_one = np.zeros((nb_clients, nb_clients))
    TV_distance_one_to_one = np.zeros((nb_clients, nb_clients))

    for i in range(nb_clients):
        KL_distance_to_average[i] = compute_KL_distance(clients[i].Y_distribution, average_client.Y_distribution)
        TV_distance_to_average[i] = compute_TV_distance(clients[i].Y_distribution, average_client.Y_distribution)

    # Compute TV distance (symmetric matrix)
    for i in range(nb_clients):
        for j in range(i, nb_clients):
            TV_distance_one_to_one[i, j] = compute_TV_distance(clients[i].Y_distribution, clients[j].Y_distribution)
            TV_distance_one_to_one[j, i] = TV_distance_one_to_one[i, j]

    # Compute KL distance
    for i in range(nb_clients):
        for j in range(nb_clients):
            KL_distance_one_to_one[i, j] = compute_KL_distance(clients[i].Y_distribution, clients[j].Y_distribution)

    return KL_distance_to_average, TV_distance_to_average, KL_distance_one_to_one, TV_distance_one_to_one


def compute_metrics_on_X(clients, average_client: Client):
    print("\n=== Compute metrics on X ===")
    nb_clients = len(clients)

    # Vector of EMD between client and the average.
    EMD_to_average = np.zeros(nb_clients)

    # Matrix of EAD between each clients
    joint_EMD = np.zeros((nb_clients, nb_clients))

    # Compute Earth Mover's distance
    print("Computing EMD with average clients ...")
    for i in tqdm(range(nb_clients)):
        EMD_to_average[i] = compute_EM_distance(clients[i].X_lower_dim, average_client.X_lower_dim)

    print("Computing EMD between clients ...")
    for i in tqdm(range(nb_clients)):
        for j in range(i, nb_clients):
            joint_EMD[i, j] = compute_EM_distance(clients[i].X_lower_dim, clients[j].X_lower_dim)
            joint_EMD[j, i] = joint_EMD[i, j]

    return EMD_to_average, joint_EMD


def compute_metrics_on_X_given_Y(clients, average_client):
    print("\n=== Compute metrics on X given Y ===")
    nb_clients = len(clients)
    nb_labels = average_client.nb_labels

    # Vector of EMD between client and the average.
    EMD_to_average = [np.zeros(nb_clients) for y in range(nb_labels)]

    # Matrix of EAD between each clients
    joint_EMD = [np.zeros((nb_clients, nb_clients)) for y in range(nb_labels)]

    for y in range(nb_labels):

        # Compute Earth Mover's distance
        print("Computing EMD with average clients ...")
        for i in tqdm(range(nb_clients)):
            EMD_to_average[y][i] = compute_EM_distance(clients[i].X_given_Y_distribution[y],
                                               average_client.X_given_Y_distribution[y])

        print("Computing EMD between clients ...")
        for i in tqdm(range(nb_clients)):
            for j in range(i, nb_clients):
                joint_EMD[y][i, j] = compute_EM_distance(clients[i].X_given_Y_distribution[y],
                                                 clients[j].X_given_Y_distribution[y])
                joint_EMD[y][j, i] = joint_EMD[y][i, j]

    return EMD_to_average, joint_EMD


def compute_metrics_on_Y_given_X(clients, average_client):
    print("\n=== Compute metrics on Y given X ===")
    nb_clients = len(clients)

    # Vector of distance between client and the average.
    KL_distance_to_average = [np.zeros(nb_clients) for x in range(NB_CLUSTER_ON_X)]
    TV_distance_to_average = [np.zeros(nb_clients) for x in range(NB_CLUSTER_ON_X)]

    # Matrix of distance between each clients
    KL_joint_distance = [np.zeros((nb_clients, nb_clients)) for x in range(NB_CLUSTER_ON_X)]
    TV_joint_distance = [np.zeros((nb_clients, nb_clients)) for x in range(NB_CLUSTER_ON_X)]

    print("\n=== Compute metrics on X given Y ===")
    for x in tqdm(range(NB_CLUSTER_ON_X)):

        for i in range(nb_clients):
            KL_distance_to_average[x][i] = compute_KL_distance(clients[i].Y_given_X_distribution[x],
                                                               average_client.Y_given_X_distribution[x])
            TV_distance_to_average[x][i] = compute_TV_distance(clients[i].Y_given_X_distribution[x],
                                                               average_client.Y_given_X_distribution[x])

        # Compute TV distance (symmetric matrix)
        for i in range(nb_clients):
            for j in range(i, nb_clients):
                TV_joint_distance[x][i, j] = compute_TV_distance(clients[i].Y_given_X_distribution[x],
                                                                 clients[j].Y_given_X_distribution[x])
                TV_joint_distance[x][j, i] = TV_joint_distance[x][i, j]

        # Compute KL distance
        for i in range(nb_clients):
            for j in range(nb_clients):
                KL_joint_distance[x][i, j] = compute_KL_distance(clients[i].Y_given_X_distribution[x],
                                                                 clients[j].Y_given_X_distribution[x])

    return KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance

