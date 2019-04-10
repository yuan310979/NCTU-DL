import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, activation_function):
        super(EEGNet, self).__init__()
        activation = None
        if activation_function == "ReLU":
            activation = nn.ReLU()
        elif activation_function == "LeakyReLU":
            activation = nn.LeakyReLU()
        elif activation_function == "ELU":
            activation = nn.ELU()

        self.firstconv = nn.Sequential(
                            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
                            nn.BatchNorm2d(16)
                         )
        self.depthwiseConv = nn.Sequential(
                                nn.Conv2d(16, 32, (2, 1), stride=(1, 1), groups=16, bias=False),
                                nn.BatchNorm2d(32),
                                activation,
                                nn.AvgPool2d((1, 4)),
                                nn.Dropout(0.25)
                             )
        self.separableConv = nn.Sequential(
                                nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0,7), bias=False),
                                nn.BatchNorm2d(32),
                                activation,
                                nn.AvgPool2d((1, 8)),
                                nn.Dropout(0.25)
                             )
        self.classify = nn.Sequential(
                            nn.Linear(736, 2)
                        )

    def forward(self, X):
        #  X = torch.Tensor(X)
        X = self.firstconv(X)
        X = self.depthwiseConv(X)
        X = self.separableConv(X)
        X = X.view(X.shape[0], -1)

        y_pred = self.classify(X)
        #  y_pred = torch.argmax(y_pred, dim=1)
        return y_pred
