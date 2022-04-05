"""Created by Constantin Philippenko, 5th April 2022."""
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from PickleHandler import pickle_saver
from Utilities import create_folder_if_not_existing


class StatisticalMetrics:

    def __init__(self, dataset_name, nb_clients) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.nb_clients = nb_clients
        self.metrics_folder = "pictures/" + dataset_name + "/metrics"
        create_folder_if_not_existing(self.metrics_folder)

    def save_itself(self):
        create_folder_if_not_existing("pickle/{0}".format(self.dataset_name))
        pickle_saver(self, "pickle/{0}/stat_metrics".format(self.dataset_name))

    def set_metrics_on_Y(self, KL_distance_to_average, TV_distance_to_average, KL_joint_distance, TV_joint_distance):
        ############## Metrics on Y ##############
        # Vector of distance between client and the average.
        self.KL_distance_to_average = KL_distance_to_average
        self.TV_distance_to_average = TV_distance_to_average

        # Matrix of distance between each clients
        self.KL_joint_distance = KL_joint_distance
        self.TV_joint_distance = TV_joint_distance

    def set_metrics_on_X(self, EMD_to_average, joint_EMD):
        ############## Metrics on X ##############
        self.EMD_to_average = EMD_to_average
        self.joint_EMD = joint_EMD

    def set_metrics_on_X_given_Y(self):
        ############## Metrics on X | Y ##############
        pass

    def set_metrics_on_Y_given_X(self):
        ############## Metrics on Y | X ##############
        pass

    def plot_Y_metrics(self):
        fig = plt.figure()

        ax1 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 0), colspan=1, rowspan=self.nb_clients)
        ax2 = plt.subplot2grid((self.nb_clients + 1, 2), (0, 1), colspan=1, rowspan=self.nb_clients)
        ax3 = plt.subplot2grid((self.nb_clients + 1, 2), (self.nb_clients, 0), colspan=1, rowspan=1)
        ax4 = plt.subplot2grid((self.nb_clients + 1, 2), (self.nb_clients, 1), colspan=1, rowspan=1)

        titles = ["TV distance for Y distribution", "KL distance for Y distribution"]
        matrix_to_plot = [self.TV_joint_distance, self.KL_joint_distance, self.TV_distance_to_average, self.KL_distance_to_average]
        axes = [ax1, ax2, ax3, ax4]

        for i in range(len(matrix_to_plot)):
            axes[i].get_yaxis().set_visible(False)
            if len(matrix_to_plot[i].shape) != 1:
                im = axes[i].imshow(matrix_to_plot[i], cmap="Blues")
                axes[i].set_title(label=titles[i])
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

        plt.savefig('{0}/{1}.eps'.format(self.metrics_folder, "Y"), format='eps')

    def plot_X_metrics(self):
        fig = plt.figure()

        ax1 = plt.subplot2grid((self.nb_clients + 1, 1), (0, 0), colspan=1, rowspan=self.nb_clients)
        ax2 = plt.subplot2grid((self.nb_clients + 1, 1), (self.nb_clients, 0), colspan=1, rowspan=1)

        titles = ["Sinkhorn distance"]
        matrix_to_plot = [self.joint_EMD, self.EMD_to_average]
        axes = [ax1, ax2]

        for i in range(len(matrix_to_plot)):
            axes[i].get_yaxis().set_visible(False)
            if len(matrix_to_plot[i].shape) != 1:
                im = axes[i].imshow(matrix_to_plot[i], cmap="Blues")
                axes[i].set_title(label=titles[i])
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

        plt.savefig('{0}/{1}.eps'.format(self.metrics_folder, "X"), format='eps')