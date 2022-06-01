"""Created by Constantin Philippenko, 5th April 2022."""
import copy
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import matplotlib
from sklearn.cluster import AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler, StandardScaler

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from src.Client import NB_CLUSTER_ON_CONTINUOUS_VAR
from src.Distance import Distance, DistanceForSeveralRuns, remove_diagonal, create_matrix_with_zeros_diagonal_from_array
from src.PickleHandler import pickle_saver
from src.Utilities import create_folder_if_not_existing

TITLES = ["IID", "NON-IID"]
COLORS = ["tab:blue", "tab:orange"]


class StatisticalMetrics:

    def __init__(self, dataset_name: str, nb_clients: int, nb_labels: int, labels_type: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.nb_clients = nb_clients
        self.nb_labels = nb_labels
        self.labels_type = labels_type
        self.metrics_folder = "pictures/" + self.dataset_name

        ############## Metrics on X ##############
        self.EM_distance_on_X = DistanceForSeveralRuns()

        ############## Metrics on Y ##############
        if self.labels_type == "discrete":
            self.KL_distance_on_Y = DistanceForSeveralRuns()
            self.TV_distance_on_Y = DistanceForSeveralRuns()
        elif self.labels_type in ["continuous", "image"]:
            self.EM_distance_on_Y = DistanceForSeveralRuns()

        ############## Metrics on Y | X ##############
        if self.labels_type == "discrete":
            self.KL_distance_on_Y_given_X = [DistanceForSeveralRuns() for i in range(NB_CLUSTER_ON_CONTINUOUS_VAR)]
            self.TV_distance_on_Y_given_X = [DistanceForSeveralRuns() for i in range(NB_CLUSTER_ON_CONTINUOUS_VAR)]
        elif self.labels_type == "continuous":
            self.EM_distance_on_Y_given_X = [DistanceForSeveralRuns() for i in range(NB_CLUSTER_ON_CONTINUOUS_VAR)]

        ############## Metrics on X | Y ##############
        # self.EM_distance_on_X_given_Y = [DistanceForSeveralRuns()  for i in range(self.nb_labels)]

        create_folder_if_not_existing(self.metrics_folder)

    def save_itself(self) -> None:
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/stat_metrics".format(self.dataset_name))

    def set_metrics_on_Y(self, metrics_on_Y) -> None:
        if self.labels_type == "discrete":
            KL_distance_on_Y, TV_distance_on_Y = metrics_on_Y
            self.KL_distance_on_Y.add_distance(KL_distance_on_Y)
            self.TV_distance_on_Y.add_distance(TV_distance_on_Y)
        elif self.labels_type in ["continuous", "image"]:
            self.EM_distance_on_Y.add_distance(metrics_on_Y)
        else:
            raise ValueError("Unrecognized labels type.")

    def set_metrics_on_X(self, EM_distance_on_X: Distance) -> None:
        self.EM_distance_on_X.add_distance(EM_distance_on_X)

    def set_metrics_on_X_given_Y(self, EM_distance_on_X_given_Y: List[Distance]) -> None:
        for y in range(self.nb_labels):
            self.EM_distance_on_X_given_Y[y].add_distance(EM_distance_on_X_given_Y[y])

    def set_metrics_on_Y_given_X(self, metrics_on_Y_given_X) -> None:
        if self.labels_type == "discrete":
            KL_distance_on_Y_given_X, TV_distance_on_Y_given_X = metrics_on_Y_given_X
            for x in range(NB_CLUSTER_ON_CONTINUOUS_VAR):
                self.KL_distance_on_Y_given_X[x].add_distance(KL_distance_on_Y_given_X[x])
                self.TV_distance_on_Y_given_X[x].add_distance(TV_distance_on_Y_given_X[x])
        elif self.labels_type == "continuous":
            for y in range(self.nb_labels):
                self.EM_distance_on_Y_given_X[y].add_distance(metrics_on_Y_given_X[y])
        else:
            raise ValueError("Unrecognized labels type.")

    def set_nb_points_by_non_iid_clients(self, nb_points_by_non_iid_clients):
        self.nb_points_by_non_iid_clients = nb_points_by_non_iid_clients

    def plot_histogram(self, distance: DistanceForSeveralRuns, suptitle: str, plot_name: str,
                       symmetric_matrix: bool = False) -> None:

        if distance.is_empty(): return
        fig, ax = plt.subplots()

        distrib_iid, distrib_non_iid = distance.get_concatenate_distance_one_to_one(symmetric_matrix)

        try:
            bins = np.histogram(np.hstack((distrib_iid, distrib_non_iid)), bins="sturges")[1]
        except:
            print(distrib_iid)
            print(distrib_non_iid)
        # sns.kdeplot(distrib_iid, label="iid", color=COLORS[0], ax=ax)
        # sns.kdeplot(distrib_non_iid, label="non-iid", color=COLORS[1], ax=ax)
        ax.hist(distrib_iid, bins, edgecolor="black", label="iid", color=COLORS[0], alpha=0.5)
        ax.hist(distrib_non_iid, bins, edgecolor="black", label="non-iid", color=COLORS[1], alpha=0.5)
        ax.set_title(suptitle, fontsize='xx-large', weight='extra bold')
        ax.set_ylabel("Value's occurrences")
        ax.legend(loc='upper right')
        plt.savefig('{0}/{1}_hist.png'.format(self.metrics_folder, plot_name), dvi=250, bbox_inches='tight')

    def plot_grouped_histogram(self, distances: List[DistanceForSeveralRuns], suptitle: str, plot_name: str, label: str,
                               symmetric_matrix: bool = False) -> None:

        if distances[0].is_empty(): return

        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
        for idx in range(len(distances)):

            distrib_iid, distrib_non_iid = distances[idx].get_concatenate_distance_one_to_one(symmetric_matrix)

            if len(np.unique(distrib_non_iid)) != 1:
                sns.kdeplot(distrib_iid, label="{0}{1}".format(label, idx), ax=axes[0], clip=(0.0, 1.0))
                sns.kdeplot(distrib_non_iid, label="{0}{1}".format(label, idx), ax=axes[1], clip=(0.0, 1.0))
            else:
                list_distrib_iid = [distances[i].get_concatenate_distance_one_to_one(symmetric_matrix)[0] for i in range(len(distances))]
                bins_iid = np.histogram(np.hstack(list_distrib_iid), bins=10)[1]
                axes[0].hist(distrib_iid, label="{0}{1}".format(label, idx), bins=bins_iid, alpha=0.5)
                list_distrib_non_iid = [distances[i].get_concatenate_distance_one_to_one(symmetric_matrix)[1] for i in
                                    range(len(distances))]
                bins_non_iid = np.histogram(np.hstack(list_distrib_non_iid), bins=10)[1]
                axes[1].hist(distrib_non_iid, label="{0}{1}".format(label, idx), bins=bins_non_iid, alpha=0.5)
        axes[0].set_title(label=TITLES[0])
        axes[1].set_title(label=TITLES[1])
        axes[0].legend(loc='best', fontsize = 5)
        axes[1].legend(loc='best', fontsize = 5)
        axes[0].set_ylabel("Density")
        fig.supxlabel("Distance")
        plt.suptitle(suptitle, fontsize ='xx-large', weight ='extra bold')
        plt.savefig('{0}/{1}_grouped_hist.png'.format(self.metrics_folder, plot_name), dvi=1000, bbox_inches='tight')

    def plot_distance(self, distance: DistanceForSeveralRuns, suptitle: str, plot_name: str, scale: bool,
                      reorder: bool = False) -> None:

        if distance.is_empty(): return

        fig, axes = plt.subplots(1, 2)

        iid_distance_one_to_one, non_iid_distance_one_to_one = distance.get_avg_distance_one_to_one()
        matrix_to_plot = [iid_distance_one_to_one,
                          non_iid_distance_one_to_one]

        if not (np.diag(matrix_to_plot[0]) < 10e-3).all():
            print("WARNING: The diagonal of the iid distance's matrix is not null")
        if not (np.diag(matrix_to_plot[1]) < 10e-3).all():
            print("WARNING: The diagonal of the non-iid distance's matrix is not null")

        if scale:
            print("Standard scaling before plotting ...")
            # matrix_to_plot = scaling(np.array([remove_diagonal(matrix_to_plot[0], symmetric_matrix=False)]).reshape(-1, 1),
            #                          matrix_to_plot, StandardScaler())
            both = np.array(remove_diagonal(matrix_to_plot[0], symmetric_matrix=False)).reshape(-1, 1)
                       # remove_diagonal(matrix_to_plot[1], symmetric_matrix=False)])).reshape(-1, 1)
            matrix_to_plot = scaling(both, matrix_to_plot, StandardScaler())

        # We clusterize the distance matrix to plot a block-matrix.
        if reorder:
            distance_threshold = 0.1 if scale else 0.2
            matrix_to_plot, clients_order = reorder_clients(matrix_to_plot, distance_threshold)
        else:
            clients_order = np.arange(self.nb_clients)

        one_to_one_min = min(min(matrix_to_plot[0].flatten()),
                             min(matrix_to_plot[1].flatten()))
        one_to_one_max = max(max(matrix_to_plot[0].flatten()),
                             max(matrix_to_plot[1].flatten()))

        # We create a custom colormap with two range of values, one for negative values (in blues), and one for positive
        # values (red).
        if scale:
            colors_neg = plt.cm.Blues(np.linspace(1, 0, 256))
            colors_pos = plt.cm.Reds(np.linspace(0, 1, 256))
            all_colors = np.vstack((colors_neg, colors_pos))
            cmap = colors.LinearSegmentedColormap.from_list(
                'terrain_map', all_colors)
            # The nan values are set to grey color. The diagonals will be set at nan.
            cmap.set_bad((204/255, 204/255, 204/255, 0.25))

            divnorm = colors.TwoSlopeNorm(vmin=one_to_one_min, vcenter=0, vmax=one_to_one_max)
            kwargs = dict(origin='lower', aspect="auto", cmap=cmap, norm=divnorm)

        else:
            # Simple colormap when values are only positive
            cmap = plt.cm.Reds
            cmap.set_bad((204/255, 204/255, 204/255, 0.25))
            kwargs = dict(origin='lower', vmin=one_to_one_min, vmax=one_to_one_max, aspect="auto", cmap=cmap)

        # We set the diagonal at nan.
        matrix_to_plot[0][np.arange(self.nb_clients), np.arange(self.nb_clients)] = np.nan
        matrix_to_plot[1][np.arange(self.nb_clients), np.arange(self.nb_clients)] = np.nan


        x_non_iid = 0
        total_nb_of_points = np.sum(self.nb_points_by_non_iid_clients)
        xticks, yticks = [], []
        for x in range(self.nb_clients):
            y_non_iid = 0
            for y in range(self.nb_clients):
                # The size of each row/column is depedent of the number of point on the client.
                data1 = matrix_to_plot[1][x:x + 1, y:y+1]
                data0 = matrix_to_plot[0][x:x + 1, y:y + 1]
                size_x = self.nb_points_by_non_iid_clients[clients_order[x]] / total_nb_of_points * self.nb_clients
                size_y = self.nb_points_by_non_iid_clients[clients_order[y]] / total_nb_of_points * self.nb_clients
                axes[0].imshow(data0, extent=[x_non_iid, x_non_iid + size_x, y_non_iid, y_non_iid + size_y],
                                     **kwargs)
                im1 = axes[1].imshow(data1, extent=[x_non_iid, x_non_iid + size_x, y_non_iid, y_non_iid + size_y],
                                     **kwargs)

                y_non_iid += size_y
                if len(yticks) != self.nb_clients:
                    yticks.append(y_non_iid+ size_y / 2)
            xticks.append(x_non_iid + size_x / 2)
            x_non_iid += size_x
        axes[0].set_title(label=TITLES[0])
        axes[1].set_title(label=TITLES[1])

        for ax in axes[:2]:
            ax.set_ylim(0, self.nb_clients)
            ax.set_xlim(0, self.nb_clients)
            ax.set_yticks(xticks)
            ax.set_xticks(xticks)
            ax.set_xticklabels(clients_order)
            ax.set_yticklabels(clients_order)
            ax.set_xlabel("Client index")

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="6%", pad=0.05)
        fig.colorbar(im1, ax=axes[1], cax=cax)

        # axes[1].get_yaxis().set_visible(False)
        plt.suptitle("{0} for {1}".format(suptitle, self.metrics_folder.split("/")[-1]), fontsize='xx-large',
                     weight='extra bold')
        plt.savefig('{0}/{1}.eps'.format(self.metrics_folder, plot_name), format='eps', bbox_inches='tight')
        np.savetxt('{0}/{1}-iid.txt'.format(self.metrics_folder, plot_name), matrix_to_plot[0], delimiter=',')
        np.savetxt('{0}/{1}-non_iid.txt'.format(self.metrics_folder, plot_name), matrix_to_plot[1], delimiter=',')

    def plot_Y_metrics(self) -> None:
        plot_name = "Y"
        print("Plot metric on ", plot_name)
        if self.labels_type == "discrete":
            self.plot_distance(self.KL_distance_on_Y, r"KL distance for ${0}$".format(plot_name),
                               "{0}_KL".format(plot_name), scale=False)
            self.plot_distance(self.TV_distance_on_Y, r"TV distance for ${0}$".format(plot_name),
                               "{0}_TV".format(plot_name), scale=False)
            self.plot_histogram(self.KL_distance_on_Y, r"KL distance for ${0}$".format(plot_name),
                                "{0}_KL".format(plot_name))
            self.plot_histogram(self.TV_distance_on_Y, r"TV distance for ${0}$".format(plot_name),
                                "{0}_TV".format(plot_name), symmetric_matrix = True)
        elif self.labels_type in ["continuous", "image"]:
            self.plot_distance(self.EM_distance_on_Y, r"Wasserstein distance for ${0}$".format(plot_name),
                               "{0}".format(plot_name), scale=True)
            self.plot_histogram(self.EM_distance_on_Y, r"Wasserstein distance for ${0}$".format(plot_name),
                                "{0}".format(plot_name), symmetric_matrix=True)
        else:
            raise ValueError("Unrecognized labels type.")

    def plot_X_metrics(self) -> None:
        plot_name = "X"
        print("Plot metric on ", plot_name)
        self.plot_distance(self.EM_distance_on_X, r"Wasserstein distance for ${0}$".format(plot_name),
                           "{0}".format(plot_name), scale=True)
        self.plot_histogram(self.EM_distance_on_X, r"Wasserstein distance for ${0}$".format(plot_name),
                            "{0}".format(plot_name), symmetric_matrix=True)

    def plot_X_given_Y_metrics(self) -> None:
        for y in range(self.nb_labels):
            plot_name = r"X|Y={0}".format(y)
            self.plot_distance(self.EM_distance_on_X_given_Y[y], r"Wasserstein distance for ${0}$".format(plot_name),
                               "{0}".format(plot_name))
            self.plot_histogram(self.EM_distance_on_X_given_Y[y], r"Wasserstein distance for ${0}$".format(plot_name),
                                "{0}".format(plot_name), symmetric_matrix=True)
        self.plot_grouped_histogram(self.EM_distance_on_X_given_Y, r"Wasserstein distance for $X|Y$", "X|Y",
                                    "$X|Y=$", symmetric_matrix=True)

    def plot_Y_given_X_metrics(self) -> None:
        if self.labels_type == "discrete":
            for x in range(NB_CLUSTER_ON_CONTINUOUS_VAR):
                plot_name = r"Y|X={0}".format(x)
                self.plot_distance(self.KL_distance_on_Y_given_X[x], r"KL distance for ${0}$".format(plot_name),
                                   "{0}_KL".format(plot_name))
                self.plot_distance(self.TV_distance_on_Y_given_X[x], r"TV distance for ${0}$".format(plot_name),
                                   "{0}_TV".format(plot_name))
                self.plot_histogram(self.KL_distance_on_Y_given_X[x], r"KL distance for ${0}$".format(plot_name),
                                    "{0}_KL".format(plot_name))
                self.plot_histogram(self.TV_distance_on_Y_given_X[x], r"TV distance for ${0}$".format(plot_name),
                                    "{0}_TV".format(plot_name), symmetric_matrix = True)

            self.plot_grouped_histogram(self.KL_distance_on_Y_given_X, r"KL distance for $Y|X$", "Y|X_KL",
                                        "$Y|X=$")
            self.plot_grouped_histogram(self.TV_distance_on_Y_given_X, r"TV distance for $Y|X$", "Y|X_TV",
                                        "$Y|X=$", symmetric_matrix=True)

        else:
            raise ValueError("Unrecognized labels type.")


def scaling(caliber, matrix_to_plot: List[np.array], scaler):
    # transform = scaler.fit_transform(caliber)
    # m0, m1 = transform[:len(transform) // 2], transform[len(transform) // 2:]
    scaler.fit(caliber)
    m0 = scaler.transform(remove_diagonal(matrix_to_plot[0], False).reshape(-1, 1))
    m1 = scaler.transform(remove_diagonal(matrix_to_plot[1], False).reshape(-1, 1))

    matrix_to_plot[0] = create_matrix_with_zeros_diagonal_from_array(m0)
    matrix_to_plot[1] = create_matrix_with_zeros_diagonal_from_array(m1)

    return matrix_to_plot


def reorder_clients(matrix_to_plot, distance_threshold):
    data_fitting = copy.deepcopy(matrix_to_plot[1])
    data_fitting[data_fitting < 0] = 0

    # Clustering the clients based on the distance matrix.
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, affinity='precomputed', linkage='average')
    agg.fit_predict(data_fitting)
    clusters = agg.labels_

    print("Clients' cluster:", clusters)
    indices_permutation = np.argsort(clusters)
    print("New clients order:", indices_permutation)
    nb_clients = len(indices_permutation)

    # Changing clients' order.
    new_matrix_to_plot = []
    for matrix in matrix_to_plot:
        new_matrix = copy.deepcopy(matrix)
        for i in range(nb_clients):
            # indices_permutation[j] goes to position j.
            for j in range(i+1, nb_clients):
                new_matrix[i, j] = matrix[indices_permutation[i], indices_permutation[j]]
                new_matrix[j, i] = new_matrix[i, j]
        new_matrix_to_plot.append(new_matrix)
    return new_matrix_to_plot, indices_permutation

