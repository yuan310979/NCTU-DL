import torch
from glob import glob

f = sorted(glob("./checkpoint/DeepConvNet/DeepConvNet_ELU*"))

for _f in f:
    result = torch.load(_f)
    print(_f, result['epoch'], result['test_acc'])
