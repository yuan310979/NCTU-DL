import torch
import argparse
import dataloader
import torchvision

from torch import nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

num_classes = 5

# Argparse
parser = argparse.ArgumentParser(description=f'Runnning Resnet Classification')
parser.add_argument('-b', '--batch-size', default=8, type=int, help="mini-batch size(default=64)")
parser.add_argument('-m', '--model', default="ResNet50_Pretrain", type=str, help="model type")
parser.add_argument('--checkpoint', type=str, help="name of checkpoint file")
parser.add_argument('--resume', type=str, help="name of checkpoint file")
args = parser.parse_args()

# DataLoader
test_dataset = dataloader.RetinopathyLoader("./data/", 'test')
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

# Model
model = None
if args.model == "ResNet18":
    print('Use ResNet18')
    model = resnet.resnet18(num_classes=5)
elif args.model == "ResNet50":
    print('Use ResNet50')
    model = resnet.resnet50(num_classes=5)
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

if torch.cuda.is_available():
    print("=> Use GPU on training...")
    model = model.cuda()

if args.resume is not None and Path(args.resume).exists():
    print(f"Weight File {args.resume} exists.")
    print(f"=> Loading checkpoint '{args.resume}'")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    #  optimizer.load_state_dict(checkpoint['optimizer'])
    best_test_acc = checkpoint['test_acc']
    print(f"Best: {best_test_acc}")

y_datas = []
y_preds = []

for test_X, test_y in tqdm(test_dataloader):
    if torch.cuda.is_available():
        test_X = test_X.cuda()
        test_y = test_y.cuda()
    y_pred = model(test_X)
    y_pred = torch.argmax(y_pred, dim=1)
    y_preds.extend(y_pred.cpu().data.numpy())
    y_datas.extend(test_y.cpu().data.numpy())

torch.save({
    'y_data': y_datas,
    'y_pred': y_preds
}, args.checkpoint)
