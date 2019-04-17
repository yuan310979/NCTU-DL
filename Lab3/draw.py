import torch
import argparse

import matplotlib.pyplot as plt


# Argparse
parser = argparse.ArgumentParser(description=f'Runnning Resnet Classification')
parser.add_argument('-m', '--model', default="pretrained", type=str, help="ResNet18 / ResNet50")
args = parser.parse_args()

train_acc = []
test_acc = []
pre_train_acc = []
pre_test_acc = []

resnet18 = torch.load("./resnet18_result")
resnet50 = torch.load("./resnet50_result")
pre_resnet18 = torch.load("./resnet18_pretrain_basic_result")
pre_resnet50 = torch.load("./resnet50_pretrain_basic_result")

if args.model == "ResNet18":
    pre_train_acc, pre_test_acc = pre_resnet18['train_acc'], pre_resnet18['test_acc']
    train_acc, test_acc = resnet18['train_acc'], resnet18['test_acc']
elif args.model == "ResNet50":
    pre_train_acc, pre_test_acc = pre_resnet50['train_acc'], pre_resnet50['test_acc']
    train_acc, test_acc = resnet50['train_acc'], resnet50['test_acc']

plt.figure(figsize=(30,24))
plt.title(f'Result Comparison({args.model})\n', fontsize=24)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy(%)', fontsize=14)
plt.plot(pre_train_acc, label='Train(with pretraining)')
plt.plot(pre_test_acc, label='Test(with pretraining)')
plt.plot(train_acc, label='Train(w/o pretraining)')
plt.plot(test_acc, label='Test(w/o pretraining)')
