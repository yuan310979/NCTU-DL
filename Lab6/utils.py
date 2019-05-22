import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

def one_hot_embedding(label, num_class):
    y = np.eye(num_class)
    return y[label]

def generate_one_hot_by_label(labels, num_class):
    return [one_hot_embedding(label, num_class) for label in labels]

def generate_random_one_hot(batch_size, num_class):
    return [one_hot_embedding(np.random.randint(num_class), num_class) for _ in range(batch_size)]

def plot_generated_image(imgs, row=2, col=2):
    idx = 0
    img_rows = [] 
    while idx != len(imgs):
        img_rows.append(torch.cat(torch.unbind(imgs[idx:idx+col]), 2))
        idx += col
    img_rows = torch.stack(img_rows)
    img_ret = torch.cat(torch.unbind(img_rows), 1) 
    return img_ret
