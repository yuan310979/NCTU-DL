import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

        self.linear_mu = nn.Linear(hidden_size, latent_size)
        self.linear_logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        mu = self.linear_mu(hidden)
        logvar = self.linear_logvar(hidden)

        return output, hidden, mu, logvar 

    def initHidden(self, c, device):
        ret = torch.zeros(1, 1, self.hidden_size-c.shape[-1], device=device)
        ret = torch.cat((ret, c), 2)
        return ret 
