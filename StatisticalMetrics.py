"""Created by Constantin Philippenko, 5th April 2022."""
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from Client import NB_CLUSTER_ON_X
from Distance import Distance, remove_diagonal, DistanceForSeveralRuns
from PickleHandler import pickle_saver
from Utilities import create_folder_if_not_existing

TITLES = ["IID", "NON-IID"]
COLORS = ["tab:blue", "tab:orange"]


class StatisticalMetrics:

    def __init__(self, dataset_name: str, nb_clients: int, nb_labels: int, iid: bool = False) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        if iid:
            self.dataset_name = "{0}-iid".format(self.dataset_name)
        self.nb_clients = nb_clients
        self.nb_labels = nb_labels
        self.metrics_folder = "pictures/" + self.dataset_name + "/metrics"

        ############## Metrics on X ##############
        self.EM_distance_on_X = DistanceForSeveralRuns()

        ############## Metrics on Y ##############
        self.KL_distance_on_Y = DistanceForSeveralRuns()
        self.TV_distance_on_Y = DistanceForSeveralRuns()

        ############## Metrics on Y | X ##############
        self.KL_distance_on_Y_given_X = [DistanceForSeveralRuns()  for i in range(NB_CLUSTER_ON_X)]
        self.TV_distance_on_Y_given_X = [DistanceForSeveralRuns() for i in range(NB_CLUSTER_ON_X)]

        ############## Metrics on X | Y ##############
        self.EM_distance_on_X_given_Y = [DistanceForSeveralRuns()  for i in range(self.nb_labels)]

        create_folder_if_not_existing(self.metrics_folder)

    def save_itself(self) -> None:
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/stat_metrics".format(self.dataset_name))

    def set_metrics_on_Y(self, KL_distance_on_Y: Distance, TV_distance_on_Y: Distance) -> None:
        self.KL_distance_on_Y.add_distance(KL_distance_on_Y)
        self.TV_distance_on_Y.add_distance(TV_distance_on_Y)

    def set_metrics_on_X(self, EM_distance_on_X: Distance) -> None:
        self.EM_distance_on_X.add_distance(EM_distance_on_X)

    def set_metrics_on_X_given_Y(self, EM_distance_on_X_given_Y: List[Distance]) -> None:
        for y in range(self.nb_labels):
            self.EM_distance_on_X_given_Y[y].add_distance(EM_distance_on_X_given_Y[y])

    def set_metrics_on_Y_given_X(self, KL_distance_on_Y_given_X: List[Distance],
                                 TV_distance_on_Y_given_X: List[Distance]) -> None:
        for x in range(NB_CLUSTER_ON_X):
            self.KL_distance_on_Y_given_X[x].add_distance(KL_distance_on_Y_given_X[x])
            self.TV_distance_on_Y_given_X[x].add_distance(TV_distance_on_Y_given_X[x])

    def plot_histogram(self, distance: DistanceForSeveralRuns, suptitle: str, plot_name: str,
                       symmetric_matrix: bool = False) -> None:
        fig, ax = plt.subplots()

        distrib_iid, distrib_non_iid = distance.get_concatenate_distance_one_to_one(symmetric_matrix)

        try:
            bins = np.histogram(np.hstack((distrib_iid, distrib_non_iid)), bins="sturges")[1]
        except:
            print(distrib_iid)
            print(distrib_non_iid)
        # sns.kdeplot(distrib_iid, label="iid", color=COLORS[0], ax=ax)
        # sns.kdeplot(distrib_non_iid, label="non-iid", color=COLORS[1], ax=ax)
        ax.hist(distrib_iid, bins, edgecolor="black", label="iid", color=COLORS[0])
        ax.hist(distrib_non_iid, bins, edgecolor="black", label="non-iid", color=COLORS[1])
        ax.set_title(suptitle, fontsize='xx-large', weight='extra bold')
        ax.set_ylabel("Value's occurrences")
        ax.legend(loc='upper right')
        plt.savefig('{0}/{1}_hist.eps'.format(self.metrics_folder, plot_name), format='eps', bbox_inches='tight')

    def plot_grouped_histogram(self, distances: List[DistanceForSeveralRuns], suptitle: str, plot_name: str, label: str,
                               symmetric_matrix: bool = False) -> None:

        # legend_line = [Line2D([0], [0], color="black", lw=1, label=label)]
        # xmax, ymax = 0, 0

        fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
        for idx in range(len(distances)):

            distrib_iid, distrib_non_iid = distances[idx].get_concatenate_distance_one_to_one(symmetric_matrix)

            # y_iid, bin_edges_iid = np.histogram(distrib_iid, bins="sturges")
            # bincenters = 0.5 * (bin_edges_iid[1:] + bin_edges_iid[:-1])
            sns.kdeplot(distrib_iid, label="{0}{1}".format(label, idx), ax=axes[0])
            # axes[0].plot(bincenters, y_iid, '-')
            # xmax = max(xmax, max(bincenters))
            # ymax = max(ymax, max(y_iid))
            # y_non_iid, bin_edges_non_iid = np.histogram(distrib_non_iid, bins=100)
            # bincenters = 0.5 * (bin_edges_non_iid[1:] + bin_edges_non_iid[:-1])
            # xmax = max(xmax, max(bincenters))
            # ymax = max(ymax, max(y_non_iid))
            sns.kdeplot(distrib_non_iid, label="{0}{1}".format(label, idx), ax=axes[1])
        axes[0].set_title(label=TITLES[0])
        axes[1].set_title(label=TITLES[1])
        axes[0].legend(loc='best', fontsize = 5)
        axes[1].legend(loc='best', fontsize = 5)
        axes[0].set_ylabel("Density")
        # plt.xlim(0 - xmax*0.05, xmax*1.1)
        # plt.ylim(0, ymax*1.1)
        fig.supxlabel("Distance")
        plt.suptitle(suptitle, fontsize ='xx-large', weight ='extra bold')
        plt.savefig('{0}/{1}_grouped_hist.eps'.format(self.metrics_folder, plot_name), format='eps', bbox_inches='tight')

    def plot_distance(self, distance: DistanceForSeveralRuns, suptitle: str, plot_name: str) -> None:

        ax1 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 0), colspan=1, rowspan=self.nb_clients)
        ax2 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 1), colspan=1, rowspan=self.nb_clients)
        ax3 = plt.subplot2grid((self.nb_clients + 1, 2), (self.nb_clients, 0), colspan=1, rowspan=1)
        ax4 = plt.subplot2grid((self.nb_clients + 1, 2), (self.nb_clients, 1), colspan=1, rowspan=1)

        axes = [ax1, ax2, ax3, ax4]

        iid_distance_one_to_one, non_iid_distance_one_to_one = distance.get_avg_distance_one_to_one()
        iid_distance_to_average, non_iid_distance_to_average = distance.get_avg_distance_to_average()
        matrix_to_plot = [iid_distance_one_to_one, non_iid_distance_one_to_one,
                          iid_distance_to_average, non_iid_distance_to_average]

        one_to_one_min = min(min(iid_distance_one_to_one.flatten()),
                             min(non_iid_distance_one_to_one.flatten()))
        one_to_one_max = max(max(iid_distance_one_to_one.flatten()),
                             max(non_iid_distance_one_to_one.flatten()))

        avg_min = min(min(iid_distance_to_average.flatten()),
                      min(non_iid_distance_to_average.flatten()))
        avg_max = max(max(iid_distance_to_average.flatten()),
                      max(non_iid_distance_to_average.flatten()))

        for i in range(len(matrix_to_plot)):
            axes[i].get_yaxis().set_visible(False)
            if len(matrix_to_plot[i].shape) != 1:
                im = axes[i].imshow(matrix_to_plot[i], cmap="Blues", vmin=one_to_one_min, vmax=one_to_one_max)
                axes[i].set_title(label=TITLES[i])
                axes[i].set_xlabel("Client index")
            else:
                im = axes[i].imshow(np.expand_dims(matrix_to_plot[i], axis=0), cmap="Blues",
                                    vmin=avg_min, vmax=avg_max)
            if i in [1, 3]:
                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(axes[i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, ax=axes[i], cax=cax)
        axes[0].get_yaxis().set_visible(True)
        axes[0].set_ylabel("Client index")
        plt.suptitle(suptitle, fontsize ='xx-large', weight ='extra bold')
        plt.savefig('{0}/{1}.eps'.format(self.metrics_folder, plot_name), format='eps', bbox_inches='tight')

    def plot_Y_metrics(self) -> None:
        plot_name = "Y"
        self.plot_distance(self.KL_distance_on_Y, r"KL distance for ${0}$".format(plot_name), "{0}_KL".format(plot_name))
        self.plot_distance(self.TV_distance_on_Y, r"TV distance for ${0}$".format(plot_name), "{0}_TV".format(plot_name))
        self.plot_histogram(self.KL_distance_on_Y, r"KL distance for ${0}$".format(plot_name),
                            "{0}_KL".format(plot_name))
        self.plot_histogram(self.TV_distance_on_Y, r"TV distance for ${0}$".format(plot_name),
                            "{0}_TV".format(plot_name), symmetric_matrix = True)

    def plot_X_metrics(self) -> None:
        plot_name = "X"
        self.plot_distance(self.EM_distance_on_X, r"Sinkhorn distance for ${0}$".format(plot_name),
                           "{0}".format(plot_name))
        self.plot_histogram(self.EM_distance_on_X, r"Sinkhorn distance for ${0}$".format(plot_name),
                            "{0}".format(plot_name), symmetric_matrix=True)

    def plot_X_given_Y_metrics(self) -> None:
        for y in range(self.nb_labels):
            plot_name = r"X|Y={0}".format(y)
            self.plot_distance(self.EM_distance_on_X_given_Y[y], r"Sinkhorn distance for ${0}$".format(plot_name),
                               "{0}".format(plot_name))
            self.plot_histogram(self.EM_distance_on_X_given_Y[y], r"Sinkhorn distance for ${0}$".format(plot_name),
                                "{0}".format(plot_name), symmetric_matrix=True)
        self.plot_grouped_histogram(self.EM_distance_on_X_given_Y, r"Sinkhorn distance for $X|Y$", "X|Y",
                                    "$X|Y=$", symmetric_matrix=True)

    def plot_Y_given_X_metrics(self) -> None:
        for x in range(NB_CLUSTER_ON_X):
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


