import torch
import argparse

import matplotlib.pyplot as plt


# Argparse
parser = argparse.ArgumentParser(description=f'Runnning Resnet Classification')
parser.add_argument('-m', '--model', default="ResNet18", type=str, help="ResNet18 / ResNet50")
args = parser.parse_args()

train_acc = []
test_acc = []
pre_train_acc = []
pre_test_acc = []
index = []

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


plt.figure(figsize=(10,6))
plt.title(f'Result Comparison({args.model})\n', fontsize=18)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy(%)', fontsize=12)
plt.plot(range(1,len(pre_train_acc)+1), pre_train_acc, label='Train(with pretraining)')
plt.plot(range(1,len(pre_test_acc)+1), pre_test_acc, label='Test(with pretraining)')
plt.plot(range(1,len(train_acc)+1), train_acc, label='Train(w/o pretraining)', marker='o')
plt.plot(range(1,len(test_acc)+1), test_acc, label='Test(w/o pretraining)', marker='o')
plt.legend(loc="upper left")
plt.grid()
plt.show()
