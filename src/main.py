"""Created by Constantin Philippenko, 5th April 2022."""
import argparse

from src.ComputeDistance import *
from src.Constants import NB_CLIENTS, NB_LABELS, OUTPUT_TYPE
from src.DataLoading import load_data, get_dataset, normalize_data
from src.StatisticalMetrics import StatisticalMetrics
from src.Utilities import print_mem_usage


# DATASET_NAME = "tcga_brca" # kits19  # tcga_brca # ixi # heart_disease


NB_RUNS = 50


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset name.",
        required=True,
    )
    args = parser.parse_args()
    dataset_name = args.dataset

    my_metrics = StatisticalMetrics(dataset_name, NB_CLIENTS[dataset_name], NB_LABELS[dataset_name],
                                    OUTPUT_TYPE[dataset_name])

    data, labels, splitted = get_dataset(dataset_name)
    print_mem_usage("Got the dataset.")
    my_metrics.set_clients_size(np.array([x.shape[0] for x in data]))

    # If the dataset has a naturel split, we need to load it only once.
    if splitted:
        non_iid_clients_network = load_data(data, labels, splitted, dataset_name, NB_CLIENTS[dataset_name],
                                            OUTPUT_TYPE[dataset_name], iid=False)

    for i in range(NB_RUNS):
        print_mem_usage("RUN {0}/{1}".format(i+1, NB_RUNS))

        ########## Regenerating data ##########
        # If the dataset has not a naturel split, we need to reload it at every run in order to randomly build clients.
        if not splitted:
            non_iid_clients_network = load_data(data, labels, splitted, dataset_name, NB_CLIENTS[dataset_name],
                                                OUTPUT_TYPE[dataset_name], iid=False)
        iid_clients_network = load_data(data, labels, splitted, dataset_name, NB_CLIENTS[dataset_name],
                                        OUTPUT_TYPE[dataset_name], iid=True)

        # ########## Compute metrics on X ##########
        EM_distance_on_X = compute_metrics_on_X(iid_clients_network, non_iid_clients_network)
        my_metrics.set_metrics_on_X(EM_distance_on_X)

        ########## Compute metrics on Y ##########
        metrics_on_Y = compute_metrics_on_Y(iid_clients_network, non_iid_clients_network, OUTPUT_TYPE[dataset_name])
        my_metrics.set_metrics_on_Y(metrics_on_Y)

        # ########## Compute metrics on Y | X ##########
        # metrics_on_Y_given_X = compute_metrics_on_Y_given_X(iid_clients_network, non_iid_clients_network,
        #                                                     LABELS_TYPE[DATASET_NAME])
        # my_metrics.set_metrics_on_Y_given_X(metrics_on_Y_given_X)

        ########## Compute metrics on X | Y ##########
        # EM_distance_on_X_given_Y = compute_metrics_on_X_given_Y(iid_clients_network, non_iid_clients_network)
        # my_metrics.set_metrics_on_X_given_Y(EM_distance_on_X_given_Y)

    if OUTPUT_TYPE[dataset_name] == "continuous":
        non_iid_clients_network.print_Y_empirical_distribution_function()

    my_metrics.set_nb_points_by_non_iid_clients(np.array([len(c.Y) for c in non_iid_clients_network.clients]))
    my_metrics.save_itself()


    my_metrics.plot_X_metrics()
    my_metrics.plot_Y_metrics()
    # my_metrics.plot_X_given_Y_metrics()
    # my_metrics.plot_Y_given_X_metrics()
