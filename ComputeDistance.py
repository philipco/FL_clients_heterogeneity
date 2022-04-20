"""Created by Constantin Philippenko, 20th April 2022."""
import math

import numpy as np
import ot
from tqdm import tqdm

from Client import Client, ClientsNetwork
from DistanceBetweenDistributions import Distance

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
        return 2 #math.nan # When distribution1 is empty.
    a, b = ot.unif(len(distrib1)), ot.unif(len(distrib2))  # uniform distribution on samples
    return ot.bregman.sinkhorn2(a, b, cost_matrix, reg=0.1)  # Wasserstein distance / EMD value


def compute_metrics_on_Y(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork):
    print("\n=== Compute metrics on Y ===")

    nb_clients = len(clients_network_iid.clients)
    KL_distance_on_Y, TV_distance_on_Y = Distance(nb_clients), Distance(nb_clients)

    for i in range(nb_clients):
        # Compute KL distance to average
        KL_distance_iid = compute_KL_distance(clients_network_iid.clients[i].Y_distribution,
                                              clients_network_iid.average_client.Y_distribution)
        KL_distance_non_iid = compute_KL_distance(clients_network_non_iid.clients[i].Y_distribution,
                                                  clients_network_non_iid.average_client.Y_distribution)
        KL_distance_on_Y.set_distance_to_average(i, KL_distance_iid, KL_distance_non_iid)

        # Compute TV distance to average
        TV_distance_iid = compute_TV_distance(clients_network_iid.clients[i].Y_distribution,
                                              clients_network_iid.average_client.Y_distribution)
        TV_distance_non_iid = compute_TV_distance(clients_network_non_iid.clients[i].Y_distribution,
                                                  clients_network_non_iid.average_client.Y_distribution)
        TV_distance_on_Y.set_distance_to_average(i, TV_distance_iid, TV_distance_non_iid)

    # Compute TV distance (symmetric matrix) one to one.
    for i in range(nb_clients):
        for j in range(i, nb_clients): # TODO i+1
            TV_distance_iid = compute_TV_distance(clients_network_iid.clients[i].Y_distribution,
                                                  clients_network_iid.clients[j].Y_distribution)
            TV_distance_non_iid = compute_TV_distance(clients_network_non_iid.clients[i].Y_distribution,
                                                      clients_network_non_iid.clients[j].Y_distribution)
            TV_distance_on_Y.set_distance_one_to_one(i, j, TV_distance_iid, TV_distance_non_iid)
            TV_distance_on_Y.set_distance_one_to_one(j, i, TV_distance_iid, TV_distance_non_iid)

    # Compute KL distance one to one.
    for i in range(nb_clients):
        for j in range(nb_clients):
            KL_distance_iid = compute_KL_distance(clients_network_iid.clients[i].Y_distribution,
                                                  clients_network_iid.clients[j].Y_distribution)
            KL_distance_non_iid = compute_KL_distance(clients_network_non_iid.clients[i].Y_distribution,
                                                      clients_network_non_iid.clients[j].Y_distribution)
            KL_distance_on_Y.set_distance_one_to_one(i, j, KL_distance_iid, KL_distance_non_iid)

    return KL_distance_on_Y, TV_distance_on_Y


def compute_metrics_on_X(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork):
    print("\n=== Compute metrics on X ===")
    nb_clients = len(clients_network_iid.clients)
    EM_distance_on_X = Distance(nb_clients)

    # Compute Earth Mover's distance
    print("Computing EMD with average clients ...")
    for i in tqdm(range(nb_clients)):
        EM_distance_iid = compute_EM_distance(clients_network_iid.clients[i].X_lower_dim,
                                              clients_network_iid.average_client.X_lower_dim)
        EM_distance_non_iid = compute_EM_distance(clients_network_non_iid.clients[i].X_lower_dim,
                                              clients_network_non_iid.average_client.X_lower_dim)
        EM_distance_on_X.set_distance_to_average(i, EM_distance_iid, EM_distance_non_iid)

    print("Computing EMD between clients ...")
    for i in tqdm(range(nb_clients)):
        for j in range(i, nb_clients):
            EM_distance_iid = compute_EM_distance(clients_network_iid.clients[i].X_lower_dim,
                                                  clients_network_iid.clients[j].X_lower_dim)
            EM_distance_non_iid = compute_EM_distance(clients_network_non_iid.clients[i].X_lower_dim,
                                                      clients_network_non_iid.clients[j].X_lower_dim)
            EM_distance_on_X.set_distance_one_to_one(i, j, EM_distance_iid, EM_distance_non_iid)
            EM_distance_on_X.set_distance_one_to_one(j, i, EM_distance_iid, EM_distance_non_iid)

    return EM_distance_on_X


def compute_metrics_on_X_given_Y(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork):
    print("\n=== Compute metrics on X given Y ===")
    nb_clients = len(clients_network_iid.clients)
    nb_labels = clients_network_non_iid.average_client.nb_labels
    EM_distance_on_X_given_Y = [Distance(nb_clients) for x in range(nb_labels)]

    for y in range(nb_labels):

        # Compute Earth Mover's distance
        print("Computing EMD with average clients ...")
        for i in tqdm(range(nb_clients)):
            EM_distance_iid = compute_EM_distance(clients_network_iid.clients[i].X_given_Y_distribution[y],
                                                  clients_network_iid.average_client.X_given_Y_distribution[y])
            EM_distance_non_iid = compute_EM_distance(clients_network_non_iid.clients[i].X_given_Y_distribution[y],
                                                      clients_network_non_iid.average_client.X_given_Y_distribution[y])
            EM_distance_on_X_given_Y[y].set_distance_to_average(i, EM_distance_iid, EM_distance_non_iid)

        print("Computing EMD between clients ...")
        for i in tqdm(range(nb_clients)):
            for j in range(i+1, nb_clients):
                EM_distance_iid = compute_EM_distance(clients_network_iid.clients[i].X_given_Y_distribution[y],
                                                      clients_network_iid.clients[j].X_given_Y_distribution[y])
                EM_distance_non_iid = compute_EM_distance(clients_network_non_iid.clients[i].X_given_Y_distribution[y],
                                                      clients_network_non_iid.clients[j].X_given_Y_distribution[y])
                EM_distance_on_X_given_Y[y].set_distance_one_to_one(i, j, EM_distance_iid, EM_distance_non_iid)
                EM_distance_on_X_given_Y[y].set_distance_one_to_one(j, i, EM_distance_iid, EM_distance_non_iid)

    return EM_distance_on_X_given_Y


def compute_metrics_on_Y_given_X(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork):
    print("\n=== Compute metrics on Y given X ===")
    nb_clients = len(clients_network_iid.clients)

    KL_distance_on_Y = [Distance(nb_clients) for x in range(NB_CLUSTER_ON_X)]
    TV_distance_on_Y = [Distance(nb_clients) for x in range(NB_CLUSTER_ON_X)]

    for x in tqdm(range(NB_CLUSTER_ON_X)):
        for i in range(nb_clients):
            # Compute KL distance to average
            KL_distance_iid = compute_KL_distance(clients_network_iid.clients[i].Y_given_X_distribution[x],
                                                  clients_network_iid.average_client.Y_given_X_distribution[x])
            KL_distance_non_iid = compute_KL_distance(clients_network_non_iid.clients[i].Y_given_X_distribution[x],
                                                      clients_network_non_iid.average_client.Y_given_X_distribution[x])
            KL_distance_on_Y[x].set_distance_to_average(i, KL_distance_iid, KL_distance_non_iid)

            # Compute TV distance to average
            TV_distance_iid = compute_TV_distance(clients_network_iid.clients[i].Y_given_X_distribution[x],
                                                  clients_network_iid.average_client.Y_given_X_distribution[x])
            TV_distance_non_iid = compute_TV_distance(clients_network_non_iid.clients[i].Y_given_X_distribution[x],
                                                      clients_network_non_iid.average_client.Y_given_X_distribution[x])
            TV_distance_on_Y[x].set_distance_to_average(i, TV_distance_iid, TV_distance_non_iid)

        # Compute TV distance (symmetric matrix) one to one.
        for i in range(nb_clients):
            for j in range(i+1, nb_clients):
                TV_distance_iid = compute_TV_distance(clients_network_iid.clients[i].Y_given_X_distribution[x],
                                                      clients_network_iid.clients[j].Y_given_X_distribution[x])
                TV_distance_non_iid = compute_TV_distance(clients_network_non_iid.clients[i].Y_given_X_distribution[x],
                                                          clients_network_non_iid.clients[j].Y_given_X_distribution[x])
                TV_distance_on_Y[x].set_distance_one_to_one(i, j, TV_distance_iid, TV_distance_non_iid)
                TV_distance_on_Y[x].set_distance_one_to_one(j, i, TV_distance_iid, TV_distance_non_iid)

        # Compute KL distance one to one.
        for i in range(nb_clients):
            for j in range(nb_clients):
                KL_distance_iid = compute_KL_distance(clients_network_iid.clients[i].Y_given_X_distribution[x],
                                                      clients_network_iid.clients[j].Y_given_X_distribution[x])
                KL_distance_non_iid = compute_KL_distance(clients_network_non_iid.clients[i].Y_given_X_distribution[x],
                                                          clients_network_non_iid.clients[j].Y_given_X_distribution[x])
                KL_distance_on_Y[x].set_distance_one_to_one(i, j, KL_distance_iid, KL_distance_non_iid)

    return KL_distance_on_Y, TV_distance_on_Y