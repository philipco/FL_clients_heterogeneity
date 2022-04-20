"""Created by Constantin Philippenko, 5th April 2022."""
import math

import ot
import torchvision.datasets as datasets

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from Client import Client, ClientsNetwork
from ComputeDistance import *
from DataLoading import load_data
from StatisticalMetrics import StatisticalMetrics

DIRICHLET_COEF = 0.8
NB_CLIENTS = 10

IID = True

DATASET_NAME = "mnist"

NB_LABELS = 10


if __name__ == '__main__':

    my_metrics = StatisticalMetrics(DATASET_NAME, NB_CLIENTS, NB_LABELS, iid=IID)
    clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=False, iid=IID)
    clients = clients_network.clients
    average_client = clients_network.average_client

    KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance = compute_metrics_on_Y(clients, average_client)
    my_metrics.set_metrics_on_Y(KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance)
    my_metrics.save_itself()
    my_metrics.plot_Y_metrics([my_metrics.TV_distance_one_to_one, my_metrics.KL_distance_one_to_one,
                               my_metrics.TV_distance_to_average, my_metrics.KL_distance_to_average], "Y")

    EMD_to_average, joint_EMD = compute_metrics_on_X(clients, average_client)
    my_metrics.set_metrics_on_X(EMD_to_average, joint_EMD)
    my_metrics.save_itself()
    my_metrics.plot_X_metrics([my_metrics.EMD_one_to_one, my_metrics.EMD_to_average], "X", "X")

    EMD_by_Y_to_average, EMD_by_Y_one_to_one = compute_metrics_on_X_given_Y(clients, average_client)
    my_metrics.set_metrics_on_X_given_Y(EMD_by_Y_to_average, EMD_by_Y_one_to_one)
    my_metrics.save_itself()
    my_metrics.plot_X_given_Y_metrics()

    KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance = compute_metrics_on_Y_given_X(clients, average_client)
    my_metrics.set_metrics_on_Y_given_X(KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance)
    my_metrics.save_itself()
    my_metrics.plot_Y_given_X_metrics()

