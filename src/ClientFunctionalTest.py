"""Created by Constantin Philippenko, 5th April 2022."""
from src.DataLoading import load_data

if __name__ == '__main__':
    NB_CLIENTS = 10

    DATASET_NAME = "mnist"

    clients_network = load_data(DATASET_NAME, NB_CLIENTS, recompute=False)

    clients_network.plot_TSNE()