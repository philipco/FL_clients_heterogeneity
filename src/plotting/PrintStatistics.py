import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind

from src.PickleHandler import pickle_loader
from src.Utilities import get_project_root, open_plotted_matrix
from src.UtilitiesNumpy import compute_entropy, remove_diagonal


def print_latex_table(datasets_names):
    # dataset_names = ["camelyon16", "ixi", "tcga_brca", "kits19", "isic2019", "heart_disease"]
    string = ""
    for name in datasets_names:
        string += name + " & "
    print(string)

    X_max, X_mean = [], []
    Y_max, Y_mean = [], []
    entropy = []

    root = get_project_root()
    for dataset_name in datasets_names:
        nb_points_by_clients = pickle_loader("{0}/pickle/{1}/processed_data/centralized".format(root, dataset_name)).nb_points_by_clients
        entropy.append(compute_entropy(nb_points_by_clients))

        distance_heter_X, distance_heter_Y = open_plotted_matrix(dataset_name)
        distance_heter_X = distance_heter_X[~np.isnan(distance_heter_X)]
        distance_heter_Y = distance_heter_Y[~np.isnan(distance_heter_Y)]
        X_max.append(np.max(distance_heter_X))
        X_mean.append(np.mean(distance_heter_X))
        Y_max.append(np.max(distance_heter_Y))
        Y_mean.append(np.mean(distance_heter_Y))
    table = np.array([X_mean, X_max, Y_mean, Y_max, entropy])
    column1 = np.array([["X mean"], ["X max"], ["Y mean"], ["Y max"], ["Entropy"]])
    table = np.append(column1, table, axis=1)
    print(tabulate(table, tablefmt="latex", floatfmt=".2f"))
    plot_metrics(X_max, X_mean, Y_max, Y_mean, entropy, datasets_name)


def plot_metrics(X_max, X_mean, Y_max, Y_mean, entropy, datasets_names):
    width = 0.25
    fontsize = 12

    fig, ax1 = plt.subplots()

    ind = np.arange(len(datasets_names))

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

    ax1.set_ylim([0, np.max(np.concatenate([X_max[1:], X_mean[1:], Y_max, Y_mean]))])

    fig.tight_layout()

    legends = [xmax, xmean, ymax, ymean, ent]
    labels = [l.get_label() for l in legends]
    plt.legend(legends, labels, loc="best", fontsize=fontsize)

    plt.xticks(ind + width / 2, datasets_name)

    plt.savefig('{0}/{1}.pdf'.format("{0}/pictures".format(get_project_root()), "comparison"), format='pdf', bbox_inches='tight')


def print_pvalue(datasets_names):
    string = ""
    for name in datasets_names:
        string += name + " & "
    print(string)

    X_pvalues, Y_pvalues = [], []
    entropy = []

    root = get_project_root()
    for dataset_name in datasets_names:
        nb_points_by_clients = pickle_loader(
            "{0}/pickle/{1}/processed_data/centralized".format(root, dataset_name)).nb_points_by_clients
        entropy.append(compute_entropy(nb_points_by_clients))

        distance_iid_X, distance_iid_Y = open_plotted_matrix(dataset_name, iid=True)
        distance_heter_X, distance_heter_Y = open_plotted_matrix(dataset_name)

        # Comment intégrer la différence de taille entre les datasets ?
        # Rajouter autant de valeurs qu'il n'y en n'a dans le set ?
        # Faire sur des sous-batch ?
        _, p = ttest_ind(remove_diagonal(distance_iid_X, True),
                         remove_diagonal(distance_heter_X, True),
                         equal_var=False)
        X_pvalues.append(p)
        _, p = ttest_ind(remove_diagonal(distance_iid_Y, True),
                         remove_diagonal(distance_heter_Y, True),
                         equal_var=False)

        Y_pvalues.append(p)

    plot_pvalues(X_pvalues, Y_pvalues, entropy, datasets_names)


def plot_pvalues(X_pvalues, Y_pvalues, entropy, datasets_names):
    width = 0.25
    fontsize = 12

    fig, ax1 = plt.subplots()

    ind = np.arange(len(datasets_names))

    greens = plt.cm.Greens
    reds = plt.cm.Reds
    blues = plt.cm.Blues

    x_pvalues = ax1.bar(ind, X_pvalues, width, label="X pvalue", lw=2, color=blues(0.8))
    y_pvalues = ax1.bar(ind + width, Y_pvalues, width, label="Y pvalue", lw=2, color=reds(0.8))
    x = np.linspace(0, len(X_pvalues))
    y = [0.05 for k in x]
    threshold = ax1.plot(x, y, color='k', label='p=0.05')

    ax2 = ax1.twinx()
    ent = ax2.bar(ind + 2*width, entropy, width, label="Entropy", lw=2, color=greens(0.6))

    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.tick_params(axis='y', labelcolor=greens(0.9), labelsize=fontsize)

    ax1.set_yscale("log")

    fig.tight_layout()

    legends = [x_pvalues, y_pvalues, ent, threshold[0]]
    labels = [l.get_label() for l in legends]
    plt.legend(legends, labels, loc="best", fontsize=fontsize)

    plt.xticks(ind + width / 2, datasets_names)

    plt.savefig('{0}/{1}.pdf'.format("{0}/pictures".format(get_project_root()), "pvalues"), format='pdf', bbox_inches='tight')

if __name__ == '__main__':

    datasets_names = ["heart_disease", "tcga_brca"]

    ### We print statistics on distance in a latex table.
    print_latex_table(datasets_names)
    print_pvalue(datasets_names)
