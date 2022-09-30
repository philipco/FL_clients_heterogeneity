"""Created by Constantin Philippenko, 29th September 2022."""
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.Constants import NB_CLIENTS


def get_dataloader(fed_dataset, train, kwargs_dataset, kwargs_dataloader):
    dataset = fed_dataset(train=train, **kwargs_dataset)
    return DataLoader(dataset, **kwargs_dataloader)


def get_element_from_dataloader(loader, X, Y):
    for x, y in loader:
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        X.append(x)
        Y.append(y)
    return X, Y


def get_data_from_pytorch(fed_dataset, kwargs_dataset, kwargs_dataloader) -> [List[np.array], List[np.array], bool]:

    # Get dataloader for train/test.
    loader_train = get_dataloader(fed_dataset, train=True, kwargs_dataset=kwargs_dataset,
                            kwargs_dataloader=kwargs_dataloader)
    loader_test = get_dataloader(fed_dataset, train=False, kwargs_dataset=kwargs_dataset,
                            kwargs_dataloader=kwargs_dataloader)

    # Get all element from the dataloader.
    X, Y = [], []
    X, Y = get_element_from_dataloader(loader_train, X, Y)
    X, Y = get_element_from_dataloader(loader_test, X, Y)

    print("Train data shape:", X[0].shape)
    print("Test data shape:", Y[0].shape)

    natural_split = False
    return [np.concatenate(X)], [np.concatenate(Y)], natural_split
