"""Created by Constantin Philippenko, 12th May 2022."""

NB_LABELS = {"mnist": 10, "fashion_mnist": 10, "camelyon16": 2, "tcga_brca": 2, "heart_disease": 2, "isic2019": 8}
LABELS_TYPE = {"mnist": "discrete", "fashion_mnist": "discrete", "camelyon16": "discrete", "tcga_brca": "continuous",
               "heart_disease": "discrete", "isic2019": "discrete"}
NB_CLIENTS = {"mnist": 10, "fashion_mnist": 10, "camelyon16": 2, "tcga_brca": 6, "heart_disease": 4, "isic2019": 6}

DEBUG = True