"""Created by Constantin Philippenko, 5th April 2022."""

from ComputeDistance import *
from DataLoading import load_data
from StatisticalMetrics import StatisticalMetrics
from Utilities import print_mem_usage

NB_CLIENTS = 10

DATASET_NAME = "mnist"

NB_LABELS = 10
NB_RUNS = 2


if __name__ == '__main__':

    my_metrics = StatisticalMetrics(DATASET_NAME, NB_CLIENTS, NB_LABELS)

    for i in range(NB_RUNS):
        print_mem_usage("RUN {0}/{1}".format(i+1, NB_RUNS))

        ########## Regenerating data ##########
        iid_clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=True, iid=True)
        non_iid_clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=True, iid=False)

        # ########## Compute metrics on X ##########
        EM_distance_on_X = compute_metrics_on_X(iid_clients_network, non_iid_clients_network)
        my_metrics.set_metrics_on_X(EM_distance_on_X)

        ########## Compute metrics on Y ##########
        # KL_distance_on_Y, TV_distance_on_Y = compute_metrics_on_Y(iid_clients_network, non_iid_clients_network)
        # my_metrics.set_metrics_on_Y(KL_distance_on_Y, TV_distance_on_Y)

        # ########## Compute metrics on Y | X ##########
        # KL_distance_on_Y_given_X, TV_distance_on_Y_given_X = compute_metrics_on_Y_given_X(iid_clients_network,
        #                                                                                   non_iid_clients_network)
        # my_metrics.set_metrics_on_Y_given_X(KL_distance_on_Y_given_X, TV_distance_on_Y_given_X)

        ########## Compute metrics on X | Y ##########
        # EM_distance_on_X_given_Y = compute_metrics_on_X_given_Y(iid_clients_network, non_iid_clients_network)
        # my_metrics.set_metrics_on_X_given_Y(EM_distance_on_X_given_Y)

    my_metrics.save_itself()

    # my_metrics.plot_Y_metrics()
    my_metrics.plot_X_metrics()
    # my_metrics.plot_X_given_Y_metrics()
    # my_metrics.plot_Y_given_X_metrics()
