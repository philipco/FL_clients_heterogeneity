"""Created by Constantin Philippenko, 5th April 2022."""

from ComputeDistance import *
from DataLoading import load_data
from StatisticalMetrics import StatisticalMetrics

DIRICHLET_COEF = 0.8
NB_CLIENTS = 10

DATASET_NAME = "mnist"

NB_LABELS = 10


if __name__ == '__main__':

    my_metrics = StatisticalMetrics(DATASET_NAME, NB_CLIENTS, NB_LABELS)

    for i in range(4):
        iid_clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=True, iid=True)
        non_iid_clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=True, iid=False)

        KL_distance_on_Y, TV_distance_on_Y = compute_metrics_on_Y(iid_clients_network, non_iid_clients_network)
        my_metrics.set_metrics_on_Y(KL_distance_on_Y, TV_distance_on_Y)

    my_metrics.save_itself()
    my_metrics.plot_Y_metrics()

    # EM_distance_on_X = compute_metrics_on_X(iid_clients_network, non_iid_clients_network)
    # my_metrics.set_metrics_on_X(EM_distance_on_X)
    # my_metrics.save_itself()
    # my_metrics.plot_X_metrics()
    #
    # EM_distance_on_X_given_Y = compute_metrics_on_X_given_Y(iid_clients_network, non_iid_clients_network)
    # my_metrics.set_metrics_on_X_given_Y(EM_distance_on_X_given_Y)
    # my_metrics.save_itself()
    # my_metrics.plot_X_given_Y_metrics()
    #
    # KL_distance_on_Y_given_X, TV_distance_on_Y_given_X = compute_metrics_on_Y_given_X(iid_clients_network, non_iid_clients_network)
    # my_metrics.set_metrics_on_Y_given_X(KL_distance_on_Y_given_X, TV_distance_on_Y_given_X)
    # my_metrics.save_itself()
    # my_metrics.plot_Y_given_X_metrics()

