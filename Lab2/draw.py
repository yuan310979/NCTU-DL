import torch 

import matplotlib.pyplot as plt

EEG_relu = None
EEG_lrelu = None
EEG_elu = None

with open("./result/EEGNet_ReLU_result", "rb") as f:
    EEG_relu = torch.load(f)
with open("./result/EEGNet_LeakyReLU_result", "rb") as f:
    EEG_lrelu = torch.load(f)
with open("./result/EEGNet_ELU_result", "rb") as f:
    EEG_elu = torch.load(f)
 
plt.figure(figsize=(30,24))
plt.title('Activation Function Comparison(EEGNet)\n', fontsize=24)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy(%)', fontsize=14)
plt.plot(EEG_relu['train_acc'], label="relu_train")
plt.plot(EEG_relu['test_acc'], label="relu_test")
plt.plot(EEG_lrelu['train_acc'], label="leaky_relu_train")
plt.plot(EEG_lrelu['test_acc'], label="leaky_relu_test")
plt.plot(EEG_elu['train_acc'], label="elu_train")
plt.plot(EEG_elu['test_acc'], label="elu_test")
plt.legend(loc="lower right")
plt.show()
