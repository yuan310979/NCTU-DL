import torch
import argparse
import dataloader
import torchvision

from torch import nn
from model import resnet
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from pathlib import Path

def save_checkpoint(state, filename):
    torch.save(state, filename)

num_classes = 5

# Argparse
parser = argparse.ArgumentParser(description=f'Runnning Resnet Classification')
parser.add_argument('-b', '--batch-size', default=8, type=int, help="mini-batch size(default=64)")
parser.add_argument('-m', '--model', default="ResNet50_Pretrain", type=str, help="model type")
parser.add_argument('--mo', default=0.9, type=float, help="momentum of optimizer")
parser.add_argument('--wd', default='5e-4', type=float, help="weight_decay(L2 penalty)")
parser.add_argument('--epochs', default=10, type=int, help="number of total epochs to run")
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help="initial learning rate")
parser.add_argument('--checkpoint', type=str, help="name of checkpoint file")
parser.add_argument('--resume', type=str, help="name of checkpoint file")
args = parser.parse_args()

# DataLoader
train_dataset = dataloader.RetinopathyLoader("./data/", 'train')
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataset = dataloader.RetinopathyLoader("./data/", 'test')
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

# Model
model = None
if args.model == "ResNet18":
    print('Use ResNet18')
    #  model = resnet.resnet18(num_classes=5)
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(num_ftrs, num_classes)
elif args.model == "ResNet50":
    print('Use ResNet50')
    #  model = resnet.resnet50(num_classes=5)
    model = torchvision.models.resnet50()
    num_ftrs = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(num_ftrs, num_classes)
elif args.model == "ResNet18_Pretrain":
    print('Use ResNet18 Pretrain')
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(num_ftrs, num_classes)
elif args.model == "ResNet50_Pretrain":
    print('Use ResNet50 Pretrain')
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(num_ftrs, num_classes)

# Construct loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
#  criterion = QWKLoss(n_classes=5)


# Use GPU to do training if available
if torch.cuda.is_available():
    print("=> Use GPU on training...")
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.mo)
#  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

# Result
train_accs = []
test_accs = []
best_test_acc = 0

if args.resume is not None and Path(args.resume).exists():
    print(f"Weight File {args.resume} exists.")
    print(f"=> Loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    #  optimizer.load_state_dict(checkpoint['optimizer'])
    best_test_acc = checkpoint['test_acc']
    print(f"Best: {best_test_acc}")

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
            train_correct = 0
            for train_X, train_y in train_dataloader:
                if torch.cuda.is_available():
                    train_X = train_X.cuda()
                    train_y = train_y.cuda()
                y_pred = model(train_X)
                y_pred = torch.argmax(y_pred, dim=1)
                train_correct += torch.sum(y_pred == train_y).item()
            total = len(train_dataset)  
            train_acc = 100 * train_correct / total
            train_accs.append(train_acc)

            # Test accuracy with respect to testing data
            test_correct = 0
            for test_X, test_y in test_dataloader:
                if torch.cuda.is_available():
                    test_X = test_X.cuda()
                    test_y = test_y.cuda()
                y_pred = model(test_X)
                y_pred = torch.argmax(y_pred, dim=1)
                test_correct += torch.sum(y_pred == test_y).item()
            total = len(test_dataset) 
            test_acc = 100 * test_correct / total
            test_accs.append(test_acc)

            if test_acc > best_test_acc:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                    'train_acc': train_acc,
                    'test_acc': test_acc
                    }, args.checkpoint)
                best_test_acc = test_acc

            epochs.set_description('[Accuracy:{:.6f} {:.6f} {:.6f}]'.format(train_acc, test_acc, best_test_acc))

            torch.save({
                'train_acc': train_accs,
                'test_acc': test_accs
                }, args.checkpoint + "_result")
