"""Created by Constantin Philippenko, 5th April 2022."""

import os
from pathlib import Path

import numpy as np
import psutil

from src.Constants import OUTPUT_TYPE


def get_project_root() -> str:
    import pathlib
    path = str(pathlib.Path().absolute())
    root_dir = str(Path(__file__).parent.parent)
    split = path.split(root_dir)
    return root_dir


def create_folder_if_not_existing(folder) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)


def file_exist(filename: str) -> None:
    return os.path.isfile(filename)


def remove_file(filename: str) -> None:
    os.remove(filename)


def print_mem_usage(info = None) -> None:
    if info:
        print(info)
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss * 10**-9 # Memory in Gb.
    print("Memory usage: {0:1.2f}Gb".format(mem))


def open_matrix(dataset_name: str) -> [np.ndarray, np.ndarray]:
    metrics_folder = "{0}/pictures/{1}".format(get_project_root(), dataset_name)
    non_iid_distance_X = np.loadtxt('{0}/{1}-non_iid.txt'.format(metrics_folder, "X"), delimiter=',')
    Y_name = "Y_TV" if OUTPUT_TYPE[dataset_name] == "discrete" else "Y"
    non_iid_distance_Y = np.loadtxt('{0}/{1}-non_iid.txt'.format(metrics_folder, Y_name), delimiter=',')
    return non_iid_distance_X, non_iid_distance_Y