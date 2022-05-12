"""Created by Constantin Philippenko, 20th April 2022."""
import math
from typing import List

import numpy as np
import ot
from tqdm import tqdm

from src.Client import ClientsNetwork, NB_CLUSTER_ON_CONTINUOUS_VAR
from src.Distance import Distance


def compute_KL_distance(distrib1: np.ndarray, distrib2: np.ndarray) -> float:
    """Kullback-Leibler."""
    assert len(distrib1) == len(distrib2), "KL: distributions are not of equal size."
    EPS = 10e-6
    N = len(distrib1)
    return np.sum([distrib1[i] * np.log((distrib1[i] / (distrib2[i]+ EPS)) + EPS) for i in range(N)])


def compute_TV_distance(distrib1: np.ndarray, distrib2: np.ndarray) -> float:
    """Total Variation."""
    assert len(distrib1) == len(distrib2), "TV: distributions are not of equal size."
    N = len(distrib1)
    return 0.5 * np.sum([np.abs(distrib1[i] - distrib2[i]) for i in range(N)])


def compute_EM_distance(distrib1: np.ndarray, distrib2: np.ndarray) -> float:
    """Earth's mover."""
    cost_matrix = ot.dist(distrib1, distrib2)
    assert len(distrib1) >= 1 and len(distrib2) >= 1, "Distributions must not be empty."
    # Sinkhorn is much faster than ot.emd2.
    # ot.emd2 throws a warning on precision.
    # TODO : what is the best choice of regularization ?
    a, b = ot.unif(len(distrib1)), ot.unif(len(distrib2))  # uniform distribution on samples
    return ot.emd2(a, b, cost_matrix)  # Wasserstein distance / EMD value


def compute_metrics_on_continuous_var(nb_clients: int, centralized_distribution: np.ndarray,
                                      iid_clients_distributions: List[np.ndarray],
                                      non_iid_clients_distributions: List[np.ndarray]) -> Distance:

    EM_distance_on_X = Distance(nb_clients)

    # Compute Earth Mover's distance
    for i in range(nb_clients):
        EM_distance_iid = compute_EM_distance(iid_clients_distributions[i],
                                              centralized_distribution)
        EM_distance_non_iid = compute_EM_distance(non_iid_clients_distributions[i],
                                                  centralized_distribution)
        EM_distance_on_X.set_distance_to_centralized(i, EM_distance_iid, EM_distance_non_iid)

    for i in range(nb_clients):
        for j in range(i, nb_clients):
            EM_distance_iid = compute_EM_distance(iid_clients_distributions[i],
                                                  iid_clients_distributions[j])
            EM_distance_non_iid = compute_EM_distance(non_iid_clients_distributions[i],
                                                      non_iid_clients_distributions[j])
            EM_distance_on_X.set_distance_one_to_one(i, j, EM_distance_iid, EM_distance_non_iid)
            EM_distance_on_X.set_distance_one_to_one(j, i, EM_distance_iid, EM_distance_non_iid)

    return EM_distance_on_X


def compute_metrics_on_discrete_var(nb_clients: int, centralized_distribution: np.ndarray,
                                    iid_clients_distributions: List[np.ndarray],
                                    non_iid_clients_distributions: List[np.ndarray]) -> [Distance, Distance]:

    KL_distance_on_Y, TV_distance_on_Y = Distance(nb_clients), Distance(nb_clients)

    for i in range(nb_clients):
        # Compute KL distance to centralized dataset
        KL_distance_iid = compute_KL_distance(iid_clients_distributions[i], centralized_distribution)
        KL_distance_non_iid = compute_KL_distance(non_iid_clients_distributions[i], centralized_distribution)
        KL_distance_on_Y.set_distance_to_centralized(i, KL_distance_iid, KL_distance_non_iid)

        # Compute TV distance to centralized dataset
        TV_distance_iid = compute_TV_distance(iid_clients_distributions[i], centralized_distribution)
        TV_distance_non_iid = compute_TV_distance(non_iid_clients_distributions[i], centralized_distribution)
        TV_distance_on_Y.set_distance_to_centralized(i, TV_distance_iid, TV_distance_non_iid)

    # Compute TV distance (symmetric matrix) one to one.
    for i in range(nb_clients):
        for j in range(i, nb_clients): # TODO i+1
            TV_distance_iid = compute_TV_distance(iid_clients_distributions[i], iid_clients_distributions[j])
            TV_distance_non_iid = compute_TV_distance(non_iid_clients_distributions[i], non_iid_clients_distributions[j])
            TV_distance_on_Y.set_distance_one_to_one(i, j, TV_distance_iid, TV_distance_non_iid)
            TV_distance_on_Y.set_distance_one_to_one(j, i, TV_distance_iid, TV_distance_non_iid)

    # Compute KL distance one to one.
    for i in range(nb_clients):
        for j in range(nb_clients):
            KL_distance_iid = compute_KL_distance(iid_clients_distributions[i], iid_clients_distributions[j])
            KL_distance_non_iid = compute_KL_distance(non_iid_clients_distributions[i], non_iid_clients_distributions[j])
            KL_distance_on_Y.set_distance_one_to_one(i, j, KL_distance_iid, KL_distance_non_iid)

    return KL_distance_on_Y, TV_distance_on_Y


def compute_metrics_on_Y(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork, labels_type: str) \
        -> [Distance, Distance]:
    print("\n=== Compute metrics on Y ===")

    nb_clients = len(clients_network_iid.clients)
    if labels_type == "discrete":
        KL, TV = compute_metrics_on_discrete_var(nb_clients, clients_network_iid.centralized.Y_distribution,
                                               [clients_network_iid.clients[i].Y_distribution for i in range(nb_clients)],
                                               [clients_network_non_iid.clients[i].Y_distribution for i in
                                                range(nb_clients)])
        return KL, TV
    elif labels_type == "continuous":
        return compute_metrics_on_continuous_var(nb_clients, clients_network_iid.centralized.Y_distribution,
                                            [clients_network_iid.clients[i].Y_distribution for i in range(nb_clients)],
                                            [clients_network_non_iid.clients[i].Y_distribution for i in
                                             range(nb_clients)])
    else:
        raise ValueError("Unrecognized labels type.")


def compute_metrics_on_Y_given_X(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork,
                                 labels_type: str) -> [Distance, Distance]:
    print("\n=== Compute metrics on Y given X ===")

    nb_clients = len(clients_network_iid.clients)
    if labels_type == "discrete":
        KL_distance_on_Y = []
        TV_distance_on_Y = []
        for x in tqdm(range(NB_CLUSTER_ON_CONTINUOUS_VAR)):
            KL, TV = compute_metrics_on_discrete_var(nb_clients, clients_network_iid.centralized.Y_distribution,
                                                   [clients_network_iid.clients[i].Y_given_X_distribution[x] for i in
                                                    range(nb_clients)],
                                                   [clients_network_non_iid.clients[i].Y_given_X_distribution[x] for i in
                                                    range(nb_clients)])
            KL_distance_on_Y.append(KL)
            TV_distance_on_Y.append(TV)
        return KL_distance_on_Y, TV_distance_on_Y
    elif labels_type == "continuous":
        EM_distance_on_Y = []
        for x in tqdm(range(NB_CLUSTER_ON_CONTINUOUS_VAR)):
            KL, TV = compute_metrics_on_continuous_var(nb_clients, clients_network_iid.centralized.Y_distribution,
                                                     [clients_network_iid.clients[i].Y_given_X_distribution[x] for i in
                                                      range(nb_clients)],
                                                     [clients_network_non_iid.clients[i].Y_given_X_distribution[x] for i
                                                      in
                                                      range(nb_clients)])
            EM_distance_on_Y.append(KL)
        return EM_distance_on_Y
    else:
        raise ValueError("Unrecognized labels type.")


def compute_metrics_on_X(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork) -> Distance:
    print("\n=== Compute metrics on X ===")
    nb_clients = len(clients_network_iid.clients)

    return compute_metrics_on_continuous_var(nb_clients, clients_network_iid.centralized.X_lower_dim,
                                                   [clients_network_iid.clients[i].X_lower_dim for i in
                                                    range(nb_clients)],
                                                   [clients_network_non_iid.clients[i].X_lower_dim for i in
                                                    range(nb_clients)])


def compute_metrics_on_X_given_Y(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork
                                 ) -> List[Distance]:
    print("\n=== Compute metrics on X given Y ===")
    nb_clients = len(clients_network_iid.clients)
    nb_labels = clients_network_non_iid.centralized.nb_labels


    EM_distance_on_X_given_Y = []
    for y in tqdm(range(nb_labels)):
        EM = compute_metrics_on_continuous_var(nb_clients, clients_network_iid.centralized.X_given_Y_distribution[y],
                                               [clients_network_iid.clients[i].X_given_Y_distribution[y] for i in
                                                range(nb_clients)],
                                                [clients_network_non_iid.clients[i].X_given_Y_distribution[y] for i
                                                in range(nb_clients)])
        EM_distance_on_X_given_Y.append(EM)
    return EM_distance_on_X_given_Y


