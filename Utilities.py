"""Created by Constantin Philippenko, 5th April 2022."""

import os
from pathlib import Path
import psutil


def get_project_root() -> str:
    import pathlib
    path = str(pathlib.Path().absolute())
    root_dir = str(Path(__file__).parent.parent.parent)
    split = path.split(root_dir)
    return split[0] + "/" + root_dir # TODO : checl that it is fine in both notebook and codes


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