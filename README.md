# FL clients heterogeneity

Repository to explore and quantify the heterogeneity in cross-silo federated learing.


## Overview

There seem to be at least two questions of interest:

1. How to measure heterogeneity between silos within a dataset.
2. How to compare heterogeneity between datasets.

There are at least two kinds of heterogeneity metrics: 
1. statistical metrics *(based primarily on the datasets)*
2. optimization metrics *(based on quantities that appear throughout the optimization process, once a loss/predictive model has been defined)*

## Code structure

The metrics are systematically computed for:
1. both iid and non-iid settings
2. w.r.t. the centralized client (the client that hold the full dataset) and between clients.

Main classes:

1. Client. Attributes: the dataset and its projection, the label, the distributions X, Y, X|Y and Y|X.
2. ClientsNetwork. Attributes: all clients, the centralized client.
3. Distance. Save the distance between clients and with the central client fo the iid and non-iid case.
4. DistanceForSeveralRuns. Attributes: list of distance computed with different dataset splits.
5. StatisticalMetrics. Attributes: the distances computed over several runs for each kind of distribution.

## Installation

``git clone https://github.com/philipco/structured_noise.git``

``conda create -c conda-forge --name FL_heter_env --file requirements.txt python=3.7``

## Usage

Run `python3 -m src.main --dataset dataset_name`.

Presently, the following dataset are accepted: 
1. camelyon16
2. lidc_idri
3. isic2019
4. tcga_brca
5. heart_disease
6. ixi
7. kits2019

## Contribution

Create a new branch : ``git checkout -b features/branch_name`` and implement all your modification on this branch. 
When the feature is ready, create a pull request on Github API, review the code, squash commits and merge on the master branch.

To run the code on a new dataset, just complete the function ``get_dataset`` in ``DataLoading.py``.

## Maintainers

[@ConstantinPhilippenko](https://github.com/philipco)

## License

[MIT](LICENSE) © Constantin Philippenko