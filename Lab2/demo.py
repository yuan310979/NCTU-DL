import torch
import argparse
import dataloader

import numpy as np

from EEGNet import EEGNet

# Argparse
parser = argparse.ArgumentParser(description=f'Runnning EEG Classification')
parser.add_argument('-a', '--activation-function', default="LeakyReLU", type=str, help="desired type of activation function")
parser.add_argument('-m', '--model', default="EEGNet", type=str, help="select model")
parser.add_argument('--checkpoint', default="./checkpoint/EEGNet_demo.pth", type=str, help="name of checkpoint file")
args = parser.parse_args()

# Prepare for data
_, _, test_X, test_y = dataloader.read_bci_data()
test_X = torch.from_numpy(test_X).float()
test_y = torch.from_numpy(np.array(test_y)).long()

# Select model
model = EEGNet(args.activation_function)

# go into evaluation mode (mainly for dropout & batch_normalization)
model.eval()

# Load weight file
model = model.cuda()
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['state_dict'])

test_X = test_X.cuda()
test_y = test_y.cuda()
y_pred = model(test_X)
y_pred = torch.argmax(y_pred, dim=1)
correct = torch.sum(y_pred == test_y).item()
total = len(test_y)
test_acc = 100 * correct / total
print(f"{test_acc}%")
