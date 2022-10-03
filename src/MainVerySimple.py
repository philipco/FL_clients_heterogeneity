"""Created by Constantin Philippenko, 29th September 2022."""

from torchvision import datasets

from src.ComputeDistance import compute_matrix_of_distances, function_to_use_to_compute_PCA_error, \
    function_to_use_to_compute_EM, function_to_use_to_compute_TV_error
from src.Constants import TRANSFORM_MIST
from src.DataLoader import get_data_from_pytorch
from src.DataProcessing import decentralized_processing_of_data, centralized_processing_of_data
from src.Metrics import Metrics
from src.plotting.PlotDistances import plot_distance

batch_size = 256

if __name__ == '__main__':

    X, Y, natural_split = get_data_from_pytorch(datasets.MNIST,
                                                kwargs_dataset = dict(root='../../DATASETS/MNIST', download=False, transform=TRANSFORM_MIST),
                                                kwargs_dataloader = dict(batch_size=batch_size, shuffle=False))

    data_centralized = centralized_processing_of_data("mnist", X, Y, natural_split, batch_size)
    data_decentralized = decentralized_processing_of_data("mnist", X, Y, natural_split, batch_size)


    metrics_PCA = Metrics("mnist", "PCA", data_decentralized.nb_of_clients, data_decentralized.nb_points_by_clients)
    metrics_TV = Metrics("mnist", "TV", data_decentralized.nb_of_clients, data_decentralized.nb_points_by_clients)
    metrics_EM = Metrics("mnist", "EM", data_centralized.nb_of_clients, data_centralized.nb_points_by_clients)

    for i in range(5):
        compute_matrix_of_distances(function_to_use_to_compute_PCA_error, data_decentralized, metrics_PCA)
        compute_matrix_of_distances(function_to_use_to_compute_EM, data_centralized, metrics_EM)
        compute_matrix_of_distances(function_to_use_to_compute_TV_error, data_centralized, metrics_TV)
        data_decentralized.resplit_iid()

    plot_distance(metrics_PCA)
    plot_distance(metrics_TV)
    plot_distance(metrics_EM)