"""Created by Constantin Philippenko, 5th April 2022."""
import numpy as np
from matplotlib import pyplot as plt
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
from DistanceBetweenDistributions import Distance
from PickleHandler import pickle_saver
from Utilities import create_folder_if_not_existing


class StatisticalMetrics:

    def __init__(self, dataset_name, nb_clients, nb_labels, iid: bool = False) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        if iid:
            self.dataset_name = "{0}-iid".format(self.dataset_name)
        self.nb_clients = nb_clients
        self.nb_labels = nb_labels
        self.metrics_folder = "pictures/" + self.dataset_name + "/metrics-TEST"
        create_folder_if_not_existing(self.metrics_folder)

    def save_itself(self):
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/stat_metrics".format(self.dataset_name))

    def set_metrics_on_Y(self, KL_distance_on_Y: Distance, TV_distance_on_Y: Distance):
        ############## Metrics on Y ##############
        # Vector of distance for TV and KL.
        self.KL_distance_on_Y = KL_distance_on_Y
        self.TV_distance_on_Y = TV_distance_on_Y

    def set_metrics_on_X(self, EMD_to_average, joint_EMD):
        ############## Metrics on X ##############
        self.EMD_to_average = EMD_to_average
        self.EMD_one_to_one = joint_EMD

    def set_metrics_on_X_given_Y(self, EMD_by_Y_to_average, EMD_by_Y_one_to_one):
        ############## Metrics on X | Y ##############
        ############## Metrics on X ##############
        self.EMD_by_Y_to_average = EMD_by_Y_to_average
        self.EMD_by_Y_one_to_one = EMD_by_Y_one_to_one

    def set_metrics_on_Y_given_X(self, KL_distance_given_X_to_average, TV_distance_given_X_to_average,
                                 KL_distance_given_X_one_to_one, TV_distance_given_X_one_to_one):
        ############## Metrics on Y | X ##############
        self.KL_distance_given_X_to_average = KL_distance_given_X_to_average
        self.TV_distance_given_X_to_average = TV_distance_given_X_to_average
        self.KL_distance_given_X_one_to_one = KL_distance_given_X_one_to_one
        self.TV_distance_given_X_one_to_one = TV_distance_given_X_one_to_one

    def plot_distance(self, distance: Distance, suptitle, plot_name):

        ax1 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 0), colspan=1, rowspan=self.nb_clients)
        ax2 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 1), colspan=1, rowspan=self.nb_clients)
        ax3 = plt.subplot2grid((self.nb_clients + 1, 2), (self.nb_clients, 0), colspan=1, rowspan=1)
        ax4 = plt.subplot2grid((self.nb_clients + 1, 2), (self.nb_clients, 1), colspan=1, rowspan=1)

        titles = ["IID", "NON-IID"]
        axes = [ax1, ax2, ax3, ax4]
        matrix_to_plot = [distance.iid_distance_one_to_one, distance.non_iid_distance_one_to_one,
                          distance.iid_distance_to_average, distance.non_iid_distance_to_average]

        one_to_one_min = min(min(distance.iid_distance_one_to_one.flatten()),
                             min(distance.non_iid_distance_one_to_one.flatten()))
        one_to_one_max = max(max(distance.iid_distance_one_to_one.flatten()),
                             max(distance.non_iid_distance_one_to_one.flatten()))

        avg_min = min(min(distance.iid_distance_to_average.flatten()),
                      min(distance.non_iid_distance_to_average.flatten()))
        avg_max = max(max(distance.iid_distance_to_average.flatten()),
                      max(distance.non_iid_distance_to_average.flatten()))

        for i in range(len(matrix_to_plot)):
            axes[i].get_yaxis().set_visible(False)
            if len(matrix_to_plot[i].shape) != 1:
                im = axes[i].imshow(matrix_to_plot[i], cmap="Blues", vmin=one_to_one_min, vmax=one_to_one_max)
                axes[i].set_title(label=titles[i])
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

        # fig, axes = plt.subplots(1, 2)
        # axes[0].hist(matrix_to_plot[0].flatten(), bins="sturges", density=True)
        # axes[0].set_title(titles[0])
        # axes[1].hist(matrix_to_plot[1].flatten(), bins="sturges", density=True)
        # axes[1].set_title(titles[1])
        # plt.title("Histogram for Y distribution")
        # plt.savefig('{0}/{1}-hist.eps'.format(self.metrics_folder, plot_name), format='eps')

    def plot_Y_metrics(self, KL_distance: Distance, TV_distance: Distance, plot_name: str = "Y"):
        self.plot_distance(KL_distance, "KL distance for {0} distribution".format(plot_name), "{0}_KL".format(plot_name))
        self.plot_distance(TV_distance, "TV distance for {0} distribution".format(plot_name), "{0}_TV".format(plot_name))


    def plot_X_metrics(self, matrix_to_plot, plot_name, title):

        ax1 = plt.subplot2grid((self.nb_clients + 1, 1), (0, 0), colspan=1, rowspan=self.nb_clients)
        ax2 = plt.subplot2grid((self.nb_clients + 1, 1), (self.nb_clients, 0), colspan=1, rowspan=1)
        axes = [ax1, ax2]

        for i in range(len(matrix_to_plot)):
            axes[i].get_yaxis().set_visible(False)
            if len(matrix_to_plot[i].shape) != 1:
                im = axes[i].imshow(matrix_to_plot[i], cmap="Blues")
                axes[i].set_title(label=r"Sinkhorn distance of {0}".format(title))
                axes[i].set_xlabel("Client index")
            else:
                im = axes[i].imshow(np.expand_dims(matrix_to_plot[i], axis=0), cmap="Blues")
            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, ax=axes[i], cax=cax)
        axes[0].get_yaxis().set_visible(True)
        axes[0].set_ylabel("Client index")

        plt.savefig('{0}/{1}.eps'.format(self.metrics_folder, plot_name), format='eps')

        plt.figure()
        plt.hist(matrix_to_plot[0].flatten(), bins = "sturges", density=True)
        plt.title("Histogram of {0}".format(title))
        plt.savefig('{0}/{1}-hist.eps'.format(self.metrics_folder, plot_name), format='eps')

    def plot_X_given_Y_metrics(self):
        for y in range(self.nb_labels):
            self.plot_X_metrics([self.EMD_by_Y_one_to_one[y], self.EMD_by_Y_to_average[y]],
                                "X_given_Y={0}".format(y), r"$X | Y = {0}$".format(y))

    def plot_Y_given_X_metrics(self):
        for x in range(NB_CLUSTER_ON_X):
            self.plot_Y_metrics([self.TV_distance_given_X_one_to_one[x], self.KL_distance_given_X_one_to_one[x],
                                 self.TV_distance_given_X_to_average[x], self.KL_distance_given_X_to_average[x]],
                                "Y_given_X={0}".format(x))
