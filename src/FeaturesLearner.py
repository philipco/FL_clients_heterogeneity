"""Created by Constantin Philippenko, 25th May 2022."""
import sys

import torch
import torch.nn as nn
import torchvision
from torch import optim, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

torch.set_printoptions(precision=2)

from src.Utilities import create_folder_if_not_existing

BATCH_SIZE = 128
DATASET_NAME = "heart_disease"


class TcgaBrcaNet(nn.Module):

    def __init__(self, input_size, output_size):
         super(TcgaBrcaNet, self).__init__()
         self.input_size = input_size
         self.l1 = nn.Linear(input_size, 4)
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(4, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class HeartDiseaseNet(nn.Module):

    def __init__(self, input_size, output_size):
         super(HeartDiseaseNet, self).__init__()
         self.input_size = input_size
         self.l1 = nn.Linear(input_size, 4)
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(4, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class MnistNet(nn.Module):

    def __init__(self, input_size, output_size):
         super(MnistNet, self).__init__()
         self.input_size = input_size
         self.l1 = nn.Linear(input_size, 16)
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x


class HeartDiseaseLoss(nn.CrossEntropyLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = torch.tensor(target.reshape(-1), dtype=torch.long)
        return super().forward(input, target)


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


def get_dataloader(dataset_name: str) -> [DataLoader, nn.Module, Optimizer, _LRScheduler, _Loss]:

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        # We reshape mnist to match with our neural network
        ReshapeTransform((-1,))

    ])
    if dataset_name == "mnist":
        from torchvision import datasets
        train_dataset = datasets.MNIST(root='../../DATASETS/MNIST', train=True, download=True, transform=transform)
        shape = 784
        model = MnistNet(shape, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.5)
        return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), model, optimizer, scheduler,\
               nn.CrossEntropyLoss()

    elif dataset_name == "fashion_mnist":
        from torchvision import datasets
        train_dataset = datasets.FashionMNIST(root='../../DATASETS', train=True, download=True, transform=transform)
        shape = 784
        model = MnistNet(shape, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.5)
        return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), model, optimizer,scheduler,\
               nn.CrossEntropyLoss()

    elif dataset_name == "camelyon16":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        from datasets.fed_camelyon16.dataset import FedCamelyon16, collate_fn
        X, Y = [], []
        train_dataset = FedCamelyon16(train=True, pooled=True, debug=DEBUG)
        return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    elif dataset_name == "tcga_brca":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby/datasets/fed_tcga_brca')
        import dataset
        # Required to import correctly lifelines
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby/datasets/')
        import lifelines
        import fed_tcga_brca
        from datasets.fed_tcga_brca.dataset import FedTcgaBrca
        from datasets.fed_tcga_brca.loss import BaselineLoss
        train_dataset = FedTcgaBrca(train=True, pooled=True)
        shape = 39
        model = TcgaBrcaNet(shape, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.4)
        return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), model, optimizer, scheduler, \
               BaselineLoss()

    elif dataset_name == "heart_disease":
        sys.path.insert(0, '/home/constantin/Github/FLamby')
        import flamby
        sys.path.insert(0, '/home/constantin/Github/FLamby/flamby')
        import datasets
        from datasets.fed_heart_disease.dataset import FedHeartDisease
        train_dataset = FedHeartDisease(train=True, pooled=True)
        shape = 16
        model = HeartDiseaseNet(shape, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), model, optimizer, scheduler, \
               HeartDiseaseLoss()

    raise ValueError("{0}: the dataset is unknown.".format(dataset_name))


def train_network(train_loader: DataLoader, model, optimizer, scheduler, criterion, dataset_name: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.to(device)

    nb_epoch = 10 if dataset_name in ["mnist", "fashion_mnist"] else 50
    nb_step = len(train_loader)
    for epoch in range(nb_epoch):
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if len(labels.shape) > 1:
                labels = labels.reshape(-1)
            if outputs.shape[0] == labels.shape[0]:
                # TCGA BRCA has no accuracy metrics.
                correct += (torch.argmax(outputs, 1) == labels).float().sum()
        print(f'Epochs [{epoch + 1}/{nb_epoch}], Losses: {loss.item():.4f}')

        if correct != 0:
            accuracy = 100 * correct / len(train_loader.dataset)
            print(f'Epochs [{epoch + 1}/{nb_epoch}], Accuracy: {accuracy.item()} %')


    model_folder = "saved_models/"
    create_folder_if_not_existing(model_folder)

    pruned_network = remove_last_layer(net)
    torch.save(pruned_network, model_folder + dataset_name + ".pt")
    return pruned_network


def remove_last_layer(net):
    return torch.nn.Sequential(*(list(net.children())[:-1]))


def features_transformation():

    dataset_name = DATASET_NAME
    train_loader, model, optimizer, scheduler, criterion = get_dataloader(dataset_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    criterion = criterion.to(device)

    trained_network = train_network(train_loader, model, optimizer, scheduler, criterion, dataset_name)

    X, Y = [], []
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        X.append(trained_network(inputs))
        Y.append(labels)

    return torch.cat(X), torch.cat(Y)



if __name__ == '__main__':

    X, Y = features_transformation()
    print(len(X))

