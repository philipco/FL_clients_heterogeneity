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
from src.Utilities import get_project_root
from src.plotting.PlotDistances import plot_distance
from src.plotting.PrintStatistics import print_latex_table, print_pvalue

# TODO
# 2. Make heartdisease/tcga_brca identical to old results.
# 3. Make isic2019 work

root = get_project_root()
FLAMBY_PATH = '{0}/../FLamby'.format(root)

sys.path.insert(0, FLAMBY_PATH)
import flamby
sys.path.insert(0, FLAMBY_PATH + '/flamby')
import datasets

from datasets.fed_heart_disease.dataset import FedHeartDisease
from datasets.fed_isic2019.dataset import FedIsic2019
from datasets.fed_tcga_brca.dataset import FedTcgaBrca

batch_size = 256
dataset_name = "tcga_brca"

if __name__ == '__main__':

    ### We the dataset naturally splitted or not.
    # X, Y, natural_split = get_data_from_pytorch(datasets.MNIST,
    #                                             kwargs_dataset = dict(root='../../DATASETS/MNIST', download=False, transform=TRANSFORM_MIST),
    #                                             kwargs_dataloader = dict(batch_size=batch_size, shuffle=False))

    X, Y, natural_split = get_data_from_flamby(FedTcgaBrca, NB_CLIENTS[dataset_name],
                                               kwargs_dataloader=dict(batch_size=batch_size, shuffle=False))

    ### We generate a non-iid datasplit if it's not already done.
    nb_of_clients = NB_CLIENTS[dataset_name]
    X, Y = create_non_iid_split(X, Y, nb_of_clients, natural_split)

    ### We process the data in a centralized or decentralized way.
    data_centralized = centralized_processing_of_data(dataset_name, X, Y, batch_size, nb_of_clients)
    data_decentralized = decentralized_processing_of_data(dataset_name, X, Y, batch_size, nb_of_clients)

    ### We define three measures : PCA and EM on features, and TV on labels.
    metrics_PCA = Metrics(dataset_name, "PCA", data_decentralized.nb_of_clients, data_decentralized.nb_points_by_clients)
    metrics_EM = Metrics(dataset_name, "EM", data_centralized.nb_of_clients, data_centralized.nb_points_by_clients)
    metrics_TV = Metrics(dataset_name, "TV", data_centralized.nb_of_clients, data_centralized.nb_points_by_clients)

    for i in range(4):
        ### We compute the distance between clients.
        compute_matrix_of_distances(function_to_use_to_compute_PCA_error, data_decentralized, metrics_PCA)
        compute_matrix_of_distances(function_to_use_to_compute_EM, data_centralized, metrics_EM)
        compute_matrix_of_distances(function_to_use_to_compute_TV_error, data_centralized, metrics_TV)

        ### We need to resplit the iid dataset.
        data_decentralized.resplit_iid()
        data_centralized.resplit_iid()

    ### We print the distances.
    plot_distance(metrics_PCA)
    plot_distance(metrics_EM)
    plot_distance(metrics_TV)

    ### We print statistics on distance in a latex table.
    # Better to it in PrintStatistics with all the datasets !
    print_latex_table([dataset_name])
    print_pvalue([dataset_name])