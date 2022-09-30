"""Created by Constantin Philippenko, 29th September 2022."""

from torchvision import datasets

from src.ComputeDistance import compute_matrix_of_distances_on_features, compute_TV_distance, \
    compute_matrix_of_distances_on_labels
from src.Constants import TRANSFORM_MIST
from src.DataLoader import get_data_from_pytorch
from src.DecentralizedProcessing import get_processed_data
from src.Metrics import Metrics
from src.plotting.PlotDistances import plot_distance

batch_size = 256

if __name__ == '__main__':

    X, Y, natural_split = get_data_from_pytorch(datasets.MNIST,
                                                kwargs_dataset = dict(root='../../DATASETS/MNIST', download=False, transform=TRANSFORM_MIST),
                                                kwargs_dataloader = dict(batch_size=batch_size, shuffle=False))

    data = get_processed_data("mnist", X, Y, natural_split)

    metrics_PCA = Metrics("mnist", "PCA", data.nb_of_clients, data.nb_points_by_clients)
    metrics_TV = Metrics("mnist", "TV", data.nb_of_clients, data.nb_points_by_clients)

    for i in range(5):
        compute_matrix_of_distances_on_features(None, data, metrics_PCA)
        compute_matrix_of_distances_on_labels(compute_TV_distance, data, metrics_TV)
        data.resplit_iid()

    plot_distance(metrics_PCA)
    plot_distance(metrics_TV)