import torch
import argparse
import dataloader

import numpy as np

from tqdm import trange
from pathlib import Path
from EEGNet import EEGNet
from DeepConvNet import  DeepConvNet
from torch.utils.data import TensorDataset, DataLoader

def save_checkpoint(state, filename):
    torch.save(state, filename)

# Argparse
parser = argparse.ArgumentParser(description=f'Runnning EEG Classification')
parser.add_argument('-a', '--activation-function', default="ReLU", type=str, help="desired type of activation function")
parser.add_argument('-b', '--batch-size', default=64, type=int, help="mini-batch size(default=64)")
parser.add_argument('-m', '--model', default="EEGNet", type=str, help="select model")
parser.add_argument('--epochs', default=300, type=int, help="number of total epochs to run")
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help="initial learning rate")
parser.add_argument('--checkpoint', type=str, help="name of checkpoint file")
args = parser.parse_args()

CHECKPOINT = args.checkpoint

# Prepare for data
train_X, train_y, test_X, test_y = dataloader.read_bci_data()
train_X = torch.from_numpy(train_X).float()
train_y = torch.from_numpy(np.array(train_y)).long()
test_X = torch.from_numpy(test_X).float()
test_y = torch.from_numpy(np.array(test_y)).long()

# DataLoader
train_dataset = TensorDataset(train_X, train_y)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

# Construct desired model
model = None
if args.model == "EEGNet":
    model = EEGNet(activation_function = args.activation_function) 
elif args.model == "DeepConvNet":
    model = DeepConvNet(activation_function=args.activation_function)

# Construct loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Load weight file if exists
if Path(CHECKPOINT).exists():
    print(f"Weight File {args.checkpoint} exists.")
    print(f"=> Loading checkpoint '{args.checkpoint}'")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Use GPU to do training if available
if torch.cuda.is_available():
    print("=> Use GPU on training...")
    model = model.cuda()
    criterion = criterion.cuda()

# Result
train_accs = []
test_accs = []
best_test_acc = 0

with trange(args.epochs) as epochs:
    for epoch in epochs:
        with trange(len(train_dataloader)) as t:
            for X_data, y_data in train_dataloader:
                if torch.cuda.is_available():
                    X_data = X_data.cuda()
                    y_data = y_data.cuda()
                y_pred = model(X_data)
                loss = criterion(y_pred, y_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_description('[loss:{:.6f}]'.format(loss.item()))
                t.update()
            # Train accuracy with respect to training data
            train_X = train_X.cuda()
            train_y = train_y.cuda()
            y_pred = model(train_X)
            y_pred = torch.argmax(y_pred, dim=1)
            correct = torch.sum(y_pred == train_y).item()
            total = train_X.shape[0]
            train_acc = 100 * correct / total
            train_accs.append(train_acc)

            # Test accuracy with respect to testing data
            test_X = test_X.cuda()
            test_y = test_y.cuda()
            y_pred = model(test_X)
            y_pred = torch.argmax(y_pred, dim=1)
            correct = torch.sum(y_pred == test_y).item()
            total = test_X.shape[0]
            test_acc = 100 * correct / total
            test_accs.append(test_acc)
            epochs.set_description('[Accuracy:{:.6f} {:.6f}]'.format(train_acc, test_acc))

            if test_acc > best_test_acc:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                    }, CHECKPOINT)
                best_test_acc = test_acc

torch.save({
    'train_acc': train_accs,
    'test_acc': test_accs
    }, CHECKPOINT + "_result")
