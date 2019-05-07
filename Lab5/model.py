import torch
import torch.nn as nn

from nn.module.encoder import EncoderRNN
from nn.module.decoder import AttnDecoderRNN

#  class CVAE(nn.Module):
    #  def __init__(self, input_size, hidden_size):
        #  super(CVAE, self).__init__()
        #  self.encoder = EncoderRNN(input_size, hidden_size) 

    #  def forward(self, x, hidden):
        #  mu, logvar = self.encoder(x, hidden) 
        #  z = reparameterize(mu, logvar)
        #  return z

    #  def reparameterize(self, mu, logvar):
        #  std = torch.exp(0.5*logvar) 
        #  eps = torch.randn_like(std)
        #  return mu + eps * std


class CVAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CVAE, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, output_size)
