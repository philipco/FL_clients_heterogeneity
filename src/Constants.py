"""Created by Constantin Philippenko, 12th May 2022."""

NB_LABELS = {"mnist": 10, "fashion_mnist": 10,
             "camelyon16": 2,  "heart_disease": 2, "isic2019": 8,
             "ixi": None, "kits19": None, "lidc_idri": None,  "tcga_brca": 2}
LABELS_TYPE = {"mnist": "discrete", "fashion_mnist": "discrete",
               "camelyon16": "discrete", "heart_disease": "discrete", "isic2019": "discrete",
               "ixi": "segmentation", "kits19": "segmentation", "lidc_idri": "segmentation", "tcga_brca": "continuous"}
NB_CLIENTS = {"mnist": 10, "fashion_mnist": 10, "camelyon16": 2, "tcga_brca": 6, "heart_disease": 4, "isic2019": 6,
              "ixi": 3}

DEBUG = True