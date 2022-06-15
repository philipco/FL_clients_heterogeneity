import shutil

from src.Utilities import create_folder_if_not_existing

if __name__ == '__main__':

    latex_folder = "../latex/heterogeneity"
    pictures_folder = "../pictures"
    create_folder_if_not_existing(latex_folder)
    for dataset_name in ["camelyon16", "heart_disease", "isic2019", "ixi", "kits19", "tcga_brca"]:
        shutil.copy("{0}/{1}/X.eps".format(pictures_folder, dataset_name),
                    "{0}/{1}-X.eps".format(latex_folder, dataset_name))
        try:
            shutil.copy("{0}/{1}/Y.eps".format(pictures_folder, dataset_name),
                        "{0}/{1}-Y.eps".format(latex_folder, dataset_name))
        except FileNotFoundError:
            shutil.copy("{0}/{1}/Y_TV.eps".format(pictures_folder, dataset_name),
                        "{0}/{1}-Y.eps".format(latex_folder, dataset_name))

