import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepConvNet(nn.Module):
    def __init__(self, activation_function):
        super(DeepConvNet, self).__init__()
        activation = None
        if activation_function == "ReLU":
            activation = nn.ReLU()
        elif activation_function == "LeakyReLU":
            activation = nn.LeakyReLU()
        elif activation_function == "ELU":
            activation = nn.ELU()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.Conv2d(25, 25, (2, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(25),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(25, 50, (1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(50),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(50, 100, (1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(100),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(100, 200, (1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(200),
            activation,
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5)
        )

        self.classify = nn.Sequential(
            nn.Linear(9200, 2)
        )

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = X.view(X.shape[0], -1)

        y_pred = self.classify(X)
        #  y_pred = torch.argmax(y_pred, dim=1)
        return y_pred
