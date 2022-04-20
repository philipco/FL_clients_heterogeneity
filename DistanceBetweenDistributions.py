"""Created by Constantin Philippenko, 20th April 2022."""
import numpy as np


class Distance:

    def __init__(self, nb_clients):
        # Vector of distance between client and the average.
        self.iid_distance_to_average = np.zeros(nb_clients)
        self.non_iid_distance_to_average = np.zeros(nb_clients)

        # Matrix of distance between each clients
        self.iid_distance_one_to_one = np.zeros((nb_clients, nb_clients))
        self.non_iid_distance_one_to_one = np.zeros((nb_clients, nb_clients))

    def set_distance_to_average(self, i, distance_iid, distance_non_iid):
        self.iid_distance_to_average[i] = distance_iid
        self.non_iid_distance_to_average[i] = distance_non_iid

    def set_distance_one_to_one(self, i, j, distance_iid, distance_non_iid):
        self.iid_distance_one_to_one[i, j] = distance_iid
        self.non_iid_distance_one_to_one[i, j] = distance_non_iid
