"""Created by Constantin Philippenko, 1st June 2022."""
import numpy as np
from matplotlib import pyplot as plt

from src.PickleHandler import pickle_loader
from src.Utilities import open_matrix, get_project_root

def plot_metrics(X_max, X_mean, Y_max, Y_mean, entropy, datasets_name):
    width = 0.25
    fontsize = 12

    fig, ax1 = plt.subplots()

    ind = np.arange(len(datasets_name))

    greens = plt.cm.Greens
    reds = plt.cm.Reds
    blues = plt.cm.Blues

    xmax = ax1.bar(ind, X_max, width, label="X max", lw=2, color=blues(0.4))
    xmean = ax1.bar(ind, X_mean, width, label="X mean", lw=2, color=blues(0.8))


    ymax = ax1.bar(ind + width, Y_max, width, label="Y max", lw=2, color=reds(0.4))
    ymean = ax1.bar(ind + width, Y_mean, width, label="Y mean", lw=2, color=reds(0.8))

    ax2 = ax1.twinx()
    ent = ax2.bar(ind + 2*width, entropy, width, label="Entropy", lw=2, color=greens(0.6))

    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.tick_params(axis='y', labelcolor=greens(0.9), labelsize=fontsize)

    fig.tight_layout()

    legends = [xmax, xmean, ymax, ymean, ent]
    labels = [l.get_label() for l in legends]
    plt.legend(legends, labels, loc="best", fontsize=fontsize)

    plt.xticks(ind + width / 2, datasets_name)

    plt.savefig('{0}/{1}.pdf'.format("{0}/pictures".format(get_project_root()), "comparison"), format='pdf', bbox_inches='tight')


def compute_entropy(clients_size: np.ndarray):
    nb_samples = np.sum(clients_size)
    clients_entropy = np.array([n / nb_samples * np.log2(n / nb_samples) for n in clients_size])
    return -np.sum(clients_entropy)

if __name__ == '__main__':

    dataset_names = ["heart_disease", "ixi", "tcga_brca"]


    X_max, X_mean = [], []
    Y_max, Y_mean = [], []
    entropy = []

    for dataset_name in dataset_names:
        clients_size = pickle_loader("{0}/pickle/{1}/stat_metrics".format(get_project_root(), dataset_name)).clients_size
        entropy.append(compute_entropy(clients_size))
        non_iid_distance_X, non_iid_distance_Y = open_matrix(dataset_name)
        non_iid_distance_X = non_iid_distance_X[~np.isnan(non_iid_distance_X)]
        non_iid_distance_Y = non_iid_distance_Y[~np.isnan(non_iid_distance_Y)]
        X_max.append(np.max(non_iid_distance_X))
        X_mean.append(np.mean(non_iid_distance_X))
        Y_max.append(np.max(non_iid_distance_Y))
        Y_mean.append(np.mean(non_iid_distance_Y))

    plot_metrics(X_max, X_mean, Y_max, Y_mean, entropy, dataset_names)
    # plot_metrics(Y_max, Y_mean, dataset_names)

