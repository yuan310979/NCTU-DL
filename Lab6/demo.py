import torch
import argparse

import torch.nn as nn
import numpy as np

from utils import generate_one_hot_by_label, plot_generated_image, generate_random_one_hot
from torchvision.utils import save_image
from model.mnist_network import _netg


def generate_noise(num, _z):
    num2idx = {0:0, 1:3, 2:2, 3:1, 4:4, 5:5, 6:6, 7:7, 8:9, 9:8}
    c = FloatTensor(generate_one_hot_by_label([num2idx[num]], 10))
    z = torch.cat((_z, c), 1)
    return z 

parser = argparse.ArgumentParser(description='InfoGan on MNIST dataset')
parser.add_argument('num', type=int, default=0)
args = parser.parse_args()

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

checkpoint = torch.load("./checkpoint/16_0.0005_0.0005_2.pth")


netg = _netg(nz=64, ngf=64, nc=1).to(device)
netg.load_state_dict(checkpoint['netg_state_dict'])

with torch.no_grad():
    imgs = []
    for i in range(10):
        _z = FloatTensor(np.random.randn(1, 54))
        z = generate_noise(args.num, _z)
        imgs.append(netg(z))
    imgs = torch.stack(imgs)
    img = plot_generated_image(imgs, 1, 10)
    save_image(img, 'demo_single_num.jpg')

    imgs = []
    _z = FloatTensor(np.random.randn(1, 54))
    for i in range(10):
        z = generate_noise(i, _z)
        imgs.append(netg(z))
    imgs = torch.stack(imgs)
    img = plot_generated_image(imgs, 1, 10)
    save_image(img, 'demo_one_to_nine.png')
