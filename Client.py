"""Created by Constantin Philippenko, 4th April 2022."""

import numpy as np


class Client:

    def __init__(self, X, Y, nb_labels) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.nb_labels = nb_labels
        self.Y_distribution = self.compute_Y_distribution()
        self.X_given_Y_distribution = self.compute_X_given_Y_distribution()

    def compute_Y_distribution(self):
        return np.array([len(self.X[self.Y == y]) / len(self.Y) for y in range(self.nb_labels)])

    def compute_X_given_Y_distribution(self):
        pass
