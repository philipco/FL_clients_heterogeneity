"""Created by Constantin Philippenko, 20th April 2022."""

import numpy as np
import math

from src.UtilitiesNumpy import remove_diagonal


class Distance:

    def __init__(self, nb_clients: int) -> None:
        # Vector of distance between clients and the centralized client.
        self.iid_distance_to_centralized = np.zeros(nb_clients)
        self.non_iid_distance_to_centralized = np.zeros(nb_clients)

        # Matrix of distance between each clients
        self.iid_distance_one_to_one = np.zeros((nb_clients, nb_clients))
        self.non_iid_distance_one_to_one = np.zeros((nb_clients, nb_clients))

    def set_distance_to_centralized(self, i: int, distance_iid: float, distance_non_iid: float) -> None:
        self.iid_distance_to_centralized[i] = distance_iid
        self.non_iid_distance_to_centralized[i] = distance_non_iid

    def set_distance_one_to_one(self, i, j, distance_iid: float, distance_non_iid: float) -> None:
        self.iid_distance_one_to_one[i, j] = distance_iid
        self.non_iid_distance_one_to_one[i, j] = distance_non_iid


class DistanceForSeveralRuns:

    def __init__(self) -> None:
        self.distance_for_various_run = []
        self.number_of_run = 0

    def is_empty(self):
        return len(self.distance_for_various_run) == 0

    def add_distance(self, distance):
        self.distance_for_various_run.append(distance)
        self.number_of_run += 1

    def get_avg_distance_to_centralized(self):
        iid = np.mean([d.iid_distance_to_centralized for d in self.distance_for_various_run], axis=0)
        non_iid = np.mean([d.non_iid_distance_to_centralized for d in self.distance_for_various_run], axis=0)
        return iid, non_iid

    def get_avg_distance_one_to_one(self):
        iid = np.mean([d.iid_distance_one_to_one for d in self.distance_for_various_run], axis=0)
        non_iid = np.mean([d.non_iid_distance_one_to_one for d in self.distance_for_various_run], axis=0)
        return iid, non_iid

    def get_concatenate_distance_one_to_one(self, symmetric_matrix: bool = False): #distances: List[Distance]) -> [np.array, np.array]:
        iid = np.concatenate([remove_diagonal(d.iid_distance_one_to_one, symmetric_matrix) for d in self.distance_for_various_run], axis=0)
        non_iid = np.concatenate([remove_diagonal(d.non_iid_distance_one_to_one, symmetric_matrix) for d in self.distance_for_various_run],axis=0)
        return iid, non_iid






