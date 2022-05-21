"""Created by Constantin Philippenko, 5th April 2022."""
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

import matplotlib
from sklearn.preprocessing import MinMaxScaler

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
        self.metrics_folder = "pictures/" + self.dataset_name + "/metrics"

        ############## Metrics on X ##############
        self.EM_distance_on_X = DistanceForSeveralRuns()

        ############## Metrics on Y ##############
        if self.labels_type == "discrete":
            self.KL_distance_on_Y = DistanceForSeveralRuns()
            self.TV_distance_on_Y = DistanceForSeveralRuns()
        elif self.labels_type == "continuous":
            self.EM_distance_on_Y = DistanceForSeveralRuns()

        ############## Metrics on Y | X ##############
        if self.labels_type == "discrete":
            self.KL_distance_on_Y_given_X = [DistanceForSeveralRuns() for i in range(NB_CLUSTER_ON_CONTINUOUS_VAR)]
            self.TV_distance_on_Y_given_X = [DistanceForSeveralRuns() for i in range(NB_CLUSTER_ON_CONTINUOUS_VAR)]
        elif self.labels_type == "continuous":
            self.EM_distance_on_Y_given_X = [DistanceForSeveralRuns() for i in range(NB_CLUSTER_ON_CONTINUOUS_VAR)]

        ############## Metrics on X | Y ##############
        self.EM_distance_on_X_given_Y = [DistanceForSeveralRuns()  for i in range(self.nb_labels)]

        create_folder_if_not_existing(self.metrics_folder)

    def save_itself(self) -> None:
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/stat_metrics".format(self.dataset_name))

    def set_metrics_on_Y(self, metrics_on_Y) -> None:
        if self.labels_type == "discrete":
            KL_distance_on_Y, TV_distance_on_Y = metrics_on_Y
            self.KL_distance_on_Y.add_distance(KL_distance_on_Y)
            self.TV_distance_on_Y.add_distance(TV_distance_on_Y)
        elif self.labels_type == "continuous":
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

    def plot_distance(self, distance: DistanceForSeveralRuns, suptitle: str, plot_name: str, scale: bool) -> None:

        if distance.is_empty(): return

        fig = plt.figure()
        ax1 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 0), colspan=1, rowspan=self.nb_clients)
        ax2 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 1), colspan=1, rowspan=self.nb_clients)

        axes = [ax1, ax2]

        iid_distance_one_to_one, non_iid_distance_one_to_one = distance.get_avg_distance_one_to_one()
        matrix_to_plot = [iid_distance_one_to_one,
                          non_iid_distance_one_to_one]

        if not (np.diag(matrix_to_plot[0]) < 10e-6).all():
            print("The diagonal of the iid distance's matrix is not null")
        if not (np.diag(matrix_to_plot[1]) < 10e-6).all():
            print("The diagonal of the non-iid distance's matrix is not null")

        if scale:
            print("Min-Max scaling before plotting ...")
            scaler = MinMaxScaler()

            scaler.fit(np.concatenate(np.array([remove_diagonal(matrix_to_plot[0], symmetric_matrix=False),
                       remove_diagonal(matrix_to_plot[1], symmetric_matrix=False)])).reshape(-1, 1))

            matrix_to_plot[0] = create_matrix_with_zeros_diagonal_from_array(
                scaler.transform(
                    remove_diagonal(matrix_to_plot[0], False).reshape(-1, 1)))

            matrix_to_plot[1] = create_matrix_with_zeros_diagonal_from_array(
                scaler.transform(
                    remove_diagonal(matrix_to_plot[1], False).reshape(-1, 1)))

        one_to_one_min = min(min(matrix_to_plot[0].flatten()),
                             min(matrix_to_plot[1].flatten()))
        one_to_one_max = max(max(matrix_to_plot[0].flatten()),
                             max(matrix_to_plot[1].flatten()))

        kwargs = dict(origin='lower', vmin=one_to_one_min, vmax=one_to_one_max, aspect="auto", cmap='Blues')
        im1 = axes[0].imshow(matrix_to_plot[0], **kwargs)
        x_non_iid = 0
        total_nb_of_points = np.sum(self.nb_points_by_non_iid_clients)
        xticks, yticks = [], []
        for x in range(self.nb_clients):
            y_non_iid = 0
            for y in range(self.nb_clients):#-1, -1, -1):
                data1 = matrix_to_plot[1][x:x + 1, y:y+1]
                size_x = self.nb_points_by_non_iid_clients[x] / total_nb_of_points * self.nb_clients
                size_y = self.nb_points_by_non_iid_clients[y] / total_nb_of_points * self.nb_clients
                im2 = axes[1].imshow(data1, extent=[x_non_iid, x_non_iid + size_x, y_non_iid, y_non_iid + size_y],
                                     **kwargs)

                y_non_iid += size_y
                if len(yticks) != self.nb_clients:
                    yticks.append(y_non_iid+ size_y / 2)
            xticks.append(x_non_iid + size_x / 2)
            x_non_iid += size_x
        axes[0].set_title(label=TITLES[0])
        axes[1].set_title(label=TITLES[1])

        # set the axis limits
        axes[1].set_ylim(0, self.nb_clients)
        axes[1].set_xlim(0, self.nb_clients)
        axes[1].set_yticks(xticks)
        axes[1].set_xticks(xticks)

        axes[0].set_xticks(np.arange(self.nb_clients))
        axes[0].set_yticks(np.arange(self.nb_clients))

        for ax in axes:
            ax.set_xticklabels([str(i) for i in range(self.nb_clients)])
            ax.set_yticklabels([str(i) for i in range(self.nb_clients)])

        axes[0].set_xlabel("Client index")
        axes[1].set_xlabel("Client index")

        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="6%", pad=0.05)
        fig.colorbar(im2, ax=axes[1], cax=cax)

        # axes[1].get_yaxis().set_visible(False)
        plt.suptitle("{0} for {1}".format(suptitle, self.metrics_folder.split("/")[-2]), fontsize='xx-large',
                     weight='extra bold')
        plt.savefig('{0}/{1}.eps'.format(self.metrics_folder, plot_name), format='eps', bbox_inches='tight')

    def plot_Y_metrics(self) -> None:
        plot_name = "Y"
        if self.labels_type == "discrete":
            self.plot_distance(self.KL_distance_on_Y, r"KL distance for ${0}$".format(plot_name),
                               "{0}_KL".format(plot_name), scale=False)
            self.plot_distance(self.TV_distance_on_Y, r"TV distance for ${0}$".format(plot_name),
                               "{0}_TV".format(plot_name), scale=False)
            self.plot_histogram(self.KL_distance_on_Y, r"KL distance for ${0}$".format(plot_name),
                                "{0}_KL".format(plot_name))
            self.plot_histogram(self.TV_distance_on_Y, r"TV distance for ${0}$".format(plot_name),
                                "{0}_TV".format(plot_name), symmetric_matrix = True)
        elif self.labels_type == "continuous":
            self.plot_distance(self.EM_distance_on_Y, r"Wasserstein distance for ${0}$".format(plot_name),
                               "{0}".format(plot_name), scale=True)
            self.plot_histogram(self.EM_distance_on_Y, r"Wasserstein distance for ${0}$".format(plot_name),
                                "{0}".format(plot_name), symmetric_matrix=True)
        else:
            raise ValueError("Unrecognized labels type.")

    def plot_X_metrics(self) -> None:
        plot_name = "X"
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


