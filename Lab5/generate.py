import torch

from model import CVAE

import glob

#  f = glob.glob("./checkpoint/*.pth")
#  for _f in sorted(f):
    #  c = torch.load(_f)
    #  print(_f, c['kl_loss'])
#  exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tense2index = {'sp': 0, 'tp': 1,'pg': 2,'p': 3}

latent_size = 32
hidden_size = 256

input_lang = torch.load("./lang_class.pth")

checkpoint = torch.load("./checkpoint/0.7902374299152759_61208.pth")
model = CVAE(28, hidden_size, latent_size, 28).to(device)
model.load_state_dict(checkpoint['state_dict'])

eps = torch.randn(1, 1, latent_size).to(device)
print(eps)
for key, val in tense2index.items():
    val = torch.tensor(val).to(device)
    target_c = model.embedding(val).view(1, 1, -1)
    z = torch.cat((eps, target_c), 2)

    decoded_idxs = model.generate(z)
    decoded_words = ''.join(input_lang.index2char[i] for i in decoded_idxs)
    print(decoded_words)
