"""Created by Constantin Philippenko, 20th April 2022."""
import math
from typing import List

import numpy as np
from numpy import linalg as LA
import ot
from tqdm import tqdm

from src.Client import ClientsNetwork, NB_CLUSTER_ON_CONTINUOUS_VAR
from src.Data import Data, DataDecentralized
from src.Distance import Distance
from src.Metrics import Metrics


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


def sub_sample(distrib, nb_sample):
    if nb_sample == distrib.shape[0]:
        return distrib
    idx = np.random.randint(distrib.shape[0], size=nb_sample)
    return distrib[idx, :]


def reconstruction_error(estimator, X, y=None):
    return LA.norm(X - estimator.inverse_transform(estimator.transform(X))) / len(X)


def compute_PCA_error(PCA_fitted, distrib: np.ndarray):
    return reconstruction_error(PCA_fitted, distrib)


def compute_matrix_of_distances_on_features(used_distance, data: DataDecentralized, metrics: Metrics) -> Metrics:

    print("Computing matrix of distances on features ...")
    # output = getattr(data, attribute_name)
    # print(output)

    # Matrix of distance between each clients
    distances_iid = np.zeros((data.nb_of_clients, data.nb_of_clients))
    distances_heter = np.zeros((data.nb_of_clients, data.nb_of_clients))
    for i in tqdm(range(data.nb_of_clients)):
        for j in range(data.nb_of_clients):
            distances_iid[i,j] = compute_PCA_error(data.PCA_fit_iid[i], data.features_iid[j])
            # distances_iid[j,i] = distances_iid[i,j]

            distances_heter[i, j] = compute_PCA_error(data.PCA_fit_heter[i], data.features_heter[j])
            # distances_heter[j, i] = distances_heter[i, j]
    metrics.add_distances(distances_iid, distances_heter)
    return metrics

def compute_matrix_of_distances_on_labels(used_distance, data: DataDecentralized, metrics: Metrics) -> Metrics:

    print("Computing matrix of distances on labels ...")
    # output = getattr(data, attribute_name)
    # print(output)

    # Matrix of distance between each clients
    distances_iid = np.zeros((data.nb_of_clients, data.nb_of_clients))
    distances_heter = np.zeros((data.nb_of_clients, data.nb_of_clients))
    for i in tqdm(range(data.nb_of_clients)):
        for j in range(data.nb_of_clients):
            distances_iid[i,j] = used_distance(data.labels_iid_distrib[i], data.labels_iid_distrib[j])
            # distances_iid[j,i] = distances_iid[i,j]

            distances_heter[i, j] = used_distance(data.labels_heter_distrib[i], data.labels_heter_distrib[j])
            # distances_heter[j, i] = distances_heter[i, j]
    metrics.add_distances(distances_iid, distances_heter)
    return metrics



def compute_EM_distance(distrib1: np.ndarray, distrib2: np.ndarray, stochastic: bool) -> float:
    """Earth's mover."""
    assert len(distrib1) >= 1 and len(distrib2) >= 1, "Distributions must not be empty."
    assert len(distrib1[0]) <= 20, "Dimension is bigger than 20."
    nb_sample1, nb_sample2 = distrib1.shape[0], distrib2.shape[0]

    percent_kept = 0.05 if stochastic else 1
    batch_size1, batch_size2 = int(nb_sample1 * percent_kept), int(percent_kept * nb_sample2)

    minibatch_emd = []

    for k in range(int(2 * percent_kept ** -1)):
        np.random.seed(k)
        sub_distrib1 = sub_sample(distrib1, batch_size1)
        np.random.seed(k)
        sub_distrib2 = sub_sample(distrib2, batch_size2)

        cost_matrix = ot.dist(sub_distrib1, sub_distrib2)
        a, b = ot.unif(len(sub_distrib1)), ot.unif(len(sub_distrib2))  # uniform distribution on samples
        # a = torch.ones(len(sub_distrib1)) / len(sub_distrib1)
        # b = torch.ones(len(sub_distrib2)) / len(sub_distrib2)
        minibatch_emd.append(ot.emd2(a, b, cost_matrix))  # Wasserstein distance / EMD value
    return np.average(minibatch_emd)
    # et si je n'averagais pas ?


def compute_metrics_on_continuous_var(nb_clients: int, centralized_distribution: np.ndarray,
                                      iid_clients_distributions: List[np.ndarray],
                                      non_iid_clients_distributions: List[np.ndarray]) -> Distance:

    EM_distance_on_X = Distance(nb_clients)
    stochastic_emd = False if max([distrib.shape[0] for distrib in non_iid_clients_distributions]) <= 1000 else True

    for i in tqdm(range(nb_clients)):
        for j in range(i, nb_clients):
            EM_distance_iid = compute_EM_distance(iid_clients_distributions[i],
                                                  iid_clients_distributions[j], stochastic_emd)
            EM_distance_non_iid = compute_EM_distance(non_iid_clients_distributions[i],
                                                      non_iid_clients_distributions[j], stochastic_emd)
            EM_distance_on_X.set_distance_one_to_one(i, j, EM_distance_iid, EM_distance_non_iid)
            EM_distance_on_X.set_distance_one_to_one(j, i, EM_distance_iid, EM_distance_non_iid)

    return EM_distance_on_X


def compute_metrics_on_discrete_var(nb_clients: int, centralized_distribution: np.ndarray,
                                    iid_clients_distributions: List[np.ndarray],
                                    non_iid_clients_distributions: List[np.ndarray]) -> [Distance, Distance]:

    KL_distance_on_Y, TV_distance_on_Y = Distance(nb_clients), Distance(nb_clients)

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


def compute_metrics_on_Y(clients_network_iid: ClientsNetwork, clients_network_non_iid: ClientsNetwork,
                         output_type: str) -> [Distance, Distance]:
    print("\n=== Compute metrics on Y ===")

    nb_clients = len(clients_network_iid.clients)
    if output_type == "discrete":
        KL, TV = compute_metrics_on_discrete_var(nb_clients, None, #clients_network_iid.centralized.Y_distribution,
                                               [clients_network_iid.clients[i].Y_distribution for i in range(nb_clients)],
                                               [clients_network_non_iid.clients[i].Y_distribution for i in
                                                range(nb_clients)])
        return KL, TV
    elif output_type in ["image", "continuous"]:
        return compute_metrics_on_continuous_var(nb_clients, None, #clients_network_iid.centralized.Y_distribution,
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
            KL, TV = compute_metrics_on_discrete_var(nb_clients, None, #clients_network_iid.centralized.Y_distribution,
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

    return compute_metrics_on_continuous_var(nb_clients, None, #clients_network_iid.centralized.X_lower_dim,
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


