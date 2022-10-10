"""Created by Constantin Philippenko, 29th September 2022."""
from future.moves import sys
from torchvision import datasets

from src.ComputeDistance import compute_matrix_of_distances, function_to_use_to_compute_PCA_error, \
    function_to_use_to_compute_EM, function_to_use_to_compute_TV_error
from src.Constants import TRANSFORM_MIST, NB_CLIENTS
from src.DataLoader import get_data_from_pytorch, get_data_from_flamby
from src.DataProcessing import decentralized_processing_of_data, centralized_processing_of_data
from src.Metrics import Metrics
from src.Split import create_non_iid_split
from src.plotting.PlotDistances import plot_distance

# TODO
# 1. Implement non-iid split with fixed size.
# 2. Make heart disease identical to old results.
# 3. Make isic2019 work

FLAMBY_PATH = '../../FLamby'

sys.path.insert(0, FLAMBY_PATH)
import flamby
# sys.path.insert(0, FLAMBY_PATH + '/flamby')
# import datasets
#
# from datasets.fed_heart_disease.dataset import FedHeartDisease
# from datasets.fed_isic2019.dataset import FedIsic2019

batch_size = 256
dataset_name = "mnist"

if __name__ == '__main__':

    ### We the dataset naturally splitted or not.
    X, Y, natural_split = get_data_from_pytorch(datasets.MNIST,
                                                kwargs_dataset = dict(root='../../DATASETS/MNIST', download=False, transform=TRANSFORM_MIST),
                                                kwargs_dataloader = dict(batch_size=batch_size, shuffle=False))

    # X, Y, natural_split = get_data_from_flamby(FedHeartDisease, NB_CLIENTS[dataset_name],
    #                                            kwargs_dataloader=dict(batch_size=batch_size, shuffle=False))

    ### We generate a non-iid datasplit if it's not already done.
    nb_of_clients = NB_CLIENTS[dataset_name]
    X, Y = create_non_iid_split(X, Y, nb_of_clients, natural_split)

    ### We process the data in a centralized or decentralized way.
    data_centralized = centralized_processing_of_data(dataset_name, X, Y, batch_size, nb_of_clients)
    data_decentralized = decentralized_processing_of_data(dataset_name, X, Y, batch_size, nb_of_clients)

    ### We define three measures : PCA and EM on features, and TV on labels.
    metrics_PCA = Metrics(dataset_name, "PCA", data_decentralized.nb_of_clients, data_decentralized.nb_points_by_clients)
    metrics_EM = Metrics(dataset_name, "EM", data_centralized.nb_of_clients, data_centralized.nb_points_by_clients)
    metrics_TV = Metrics(dataset_name, "TV", data_decentralized.nb_of_clients, data_decentralized.nb_points_by_clients)

    for i in range(50):
        ### We compute the distance between clients.
        compute_matrix_of_distances(function_to_use_to_compute_PCA_error, data_decentralized, metrics_PCA)
        compute_matrix_of_distances(function_to_use_to_compute_EM, data_centralized, metrics_EM)
        compute_matrix_of_distances(function_to_use_to_compute_TV_error, data_centralized, metrics_TV)

        ### We need to resplit the iid dataset.
        data_decentralized.resplit_iid()

    ### We print the distances.
    plot_distance(metrics_PCA)
    plot_distance(metrics_TV)
    plot_distance(metrics_EM)