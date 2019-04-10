import torch 

import matplotlib.pyplot as plt

DCN_relu = None
DCN_lrelu = None
DCN_elu = None

with open("./result/DeepConvNet/DeepConvNet_ReLU_2048_0.001_result", "rb") as f:
    DCN_relu = torch.load(f)
with open("./result/DeepConvNet/DeepConvNet_LeakyReLU_2048_0.001_result", "rb") as f:
    DCN_lrelu = torch.load(f)
with open("./result/DeepConvNet/DeepConvNet_ELU_2048_0.001_result", "rb") as f:
    DCN_elu = torch.load(f)
 
plt.figure(figsize=(30,24))
plt.title('Activation Function Comparison(DeepConvNet)\n', fontsize=24)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy(%)', fontsize=14)
plt.plot(DCN_relu['train_acc'], label="relu_train")
plt.plot(DCN_relu['test_acc'], label="relu_test")
plt.plot(DCN_lrelu['train_acc'], label="leaky_relu_train")
plt.plot(DCN_lrelu['test_acc'], label="leaky_relu_test")
plt.plot(DCN_elu['train_acc'], label="elu_train")
plt.plot(DCN_elu['test_acc'], label="elu_test")
plt.legend(loc="lower right")
plt.show()
