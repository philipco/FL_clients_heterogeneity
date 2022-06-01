"""Created by Constantin Philippenko, 1st June 2022."""
import numpy as np
from matplotlib import pyplot as plt

from src.Utilities import open_matrix

colors=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

def plot_metrics(X_max, X_mean, Y_max, Y_mean, datasets_name):
    width = 0.3

    fig, ax = plt.subplots(1, 1)
    ind = np.arange(len(datasets_name))


    plt.bar(ind + width, X_max, width, label="X mean", lw=2, color=colors[1])
    plt.bar(ind + width, X_mean, width, label="X max", lw=2, color=colors[0])


    plt.bar(ind, Y_max, width, label="Y mean", lw=2, color=colors[3])
    plt.bar(ind, Y_mean, width, label="Y max", lw=2, color=colors[2])



    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width / 2, datasets_name, fontsize=15)

    # Finding the best position for legends and putting it
    plt.legend(loc='best')


    # ax.set_xticklabels(datasets_name, fontsize=15)
    # ax.set_xticks(np.arange(len(datasets_name)))

    plt.legend()
    plt.show()


if __name__ == '__main__':

    dataset_names = ["heart_disease", "isic2019", "ixi", "tcga_brca"]

    X_max, X_mean = [], []
    Y_max, Y_mean = [], []

    for dataset_name in dataset_names:
        non_iid_distance_X, non_iid_distance_Y = open_matrix(dataset_name)
        non_iid_distance_X = non_iid_distance_X[~np.isnan(non_iid_distance_X)]
        non_iid_distance_Y = non_iid_distance_Y[~np.isnan(non_iid_distance_Y)]
        X_max.append(np.max(non_iid_distance_X))
        X_mean.append(np.mean(non_iid_distance_X))
        Y_max.append(np.max(non_iid_distance_Y))
        Y_mean.append(np.mean(non_iid_distance_Y))

    plot_metrics(X_max, X_mean, Y_max, Y_mean, dataset_names)
    # plot_metrics(Y_max, Y_mean, dataset_names)

