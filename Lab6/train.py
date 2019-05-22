import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np

from tqdm import trange
from utils import generate_one_hot_by_label, plot_generated_image, generate_random_one_hot
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model.mnist_network import _netg, _netd
from tensorboardX import SummaryWriter

plot_every = 100

# unix-like command line setting 
parser = argparse.ArgumentParser(description='InfoGan on MNIST dataset')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                metavar='n',
                help='mini-batch size (default: 8), this is the total '
                     'batch size of all gpus on the current node when '
                     'using data parallel or distributed data parallel')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='n',
                help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='n',
                help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='n',
                help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
        metavar='lr', help='initial learning rate (default: 5e-4)', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                metavar='w', help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default='', type=str, metavar='path',
                help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=0,
                help='set gpu number')
args = parser.parse_args()

# TensorBoard Settings
writer = SummaryWriter()

# Cuda Settings
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

dataset = datasets.MNIST('./MNIST', train=True, transform=transform, target_transform=None, download=True)
dataloder = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

netg = _netg(nz=64, ngf=64, nc=1).to(device)
netd = _netd(ndf=64, nc=1).to(device)

criterionD = nn.BCELoss()
criterionQ = nn.CrossEntropyLoss()

optimizerD = optim.Adam(netd.parameters())
optimizerG = optim.Adam(netg.parameters())
optimizerInfo = optim.Adam([{'params': netg.parameters()}, {'params': netd.parameters()}])

plot = 1

with trange(args.start_epoch, args.epochs) as t:
    for epoch in t:
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_info_loss = 0.0
        total_real_prob = 0.0
        total_fake_prob = 0.0
        total_fake2_prob = 0.0

        netg.train()
        with trange(len(dataloder)) as t2:
            for i, (X, y) in enumerate(dataloder):
                batch_size = X.shape[0]

                # Move variable to cuda
                X = X.to(device)
                y = y.to(device)

                # Discriminator ground truth
                fake_label = FloatTensor(np.zeros((batch_size, 1)))
                real_label = FloatTensor(np.ones((batch_size, 1)))

                """
                Train Generator
                """
                # Generate noise _z, label condition c and concatenated noise z
                _z = FloatTensor(np.random.randn(batch_size, 54))
                c = FloatTensor(generate_random_one_hot(batch_size, 10))
                z = torch.cat((_z, c), 1)

                optimizerG.zero_grad()

                # Generate images
                fake_image = netg(z)
                fake = netd.forward_d(fake_image)
                g_loss = criterionD(fake, real_label)

                g_loss.backward()
                optimizerG.step()

                """
                Train Discriminator
                """
                # Generate noise _z, label condition c and concatenated noise z
                _z = FloatTensor(np.random.randn(batch_size, 54))
                c = FloatTensor(generate_random_one_hot(batch_size, 10))
                z = torch.cat((_z, c), 1)

                optimizerD.zero_grad()

                fake_image2 = netg(z)
                fake2 = netd.forward_d(fake_image2.detach()) 
                fake_loss = criterionD(fake2, fake_label)
                real = netd.forward_d(X)
                real_loss = criterionD(real, real_label)
                d_loss = (fake_loss + real_loss) / 2
                
                d_loss.backward()
                optimizerD.step()

                """
                Train Generator & Q-network via InfoLoss
                """
                # Generate noise _z, label condition c and concatenated noise z
                _z = FloatTensor(np.random.randn(batch_size, 54))
                c = FloatTensor(generate_one_hot_by_label(y, 10))
                z = torch.cat((_z, c), 1)

                fake_image3 = netg(z)

                optimizerInfo.zero_grad()

                Q_c = netd.forward_Q(fake_image3) 
                info_loss = criterionQ(Q_c, y)

                info_loss.backward()
                optimizerInfo.step()
                
                total_g_loss += g_loss.data / batch_size
                total_d_loss += d_loss.data / batch_size
                total_info_loss += info_loss.data /batch_size

                total_real_prob += real.mean(0).data
                total_fake_prob += fake.mean(0).data
                total_fake2_prob += fake2.mean(0).data
                t2.update()

                if i % plot_every == 0 and i != 0:
                    writer.add_scalars('infogan/loss', {'g_loss': total_g_loss / plot_every,
                                                        'd_loss': total_d_loss / plot_every,
                                                        'info_loss': total_info_loss / plot_every}, plot)

                    writer.add_scalars('infogan/prob', {'real_data': total_real_prob / plot_every,
                                                        'fake_data_before': total_fake_prob / plot_every,
                                                        'fake_data_after': total_fake2_prob / plot_every}, plot)
                    plot += 1
                    total_g_loss = 0.0
                    total_d_loss = 0.0
                    total_info_loss = 0.0
                    total_real_prob = 0.0
                    total_fake_prob = 0.0
                    total_fake2_prob = 0.0

            t.update()

            with torch.no_grad():
                imgs = []
                netg.eval()
                for row in range(10):
                    _z = FloatTensor(np.random.randn(1, 54))
                    for y in range(10):
                        c = FloatTensor(generate_one_hot_by_label([y], 10))
                        z = torch.cat((_z, c), 1)
                        imgs.append(netg(z).squeeze(1))
                imgs = torch.stack(imgs)
                img = plot_generated_image(imgs, 10, 10)
                writer.add_image('Generate Image', img, epoch)
