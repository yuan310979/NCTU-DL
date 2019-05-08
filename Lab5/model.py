import torch
import torch.nn as nn
import random

from nn.module.encoder import EncoderRNN
from nn.module.decoder import DecoderRNN

max_length = 20
SOS_token = 0
EOS_token = 1

class CVAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, output_size, teacher_forcing_ratio=0.5):
        super(CVAE, self).__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.embedding = nn.Embedding(4, 8)
        self.encoder = EncoderRNN(input_size, hidden_size, latent_size)
        self.fc = nn.Linear(latent_size+8, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_size)

    def forward(self, input_tensor, input_c, target_c):
        input_c = self.embedding(input_c)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder_hidden = self.encoder.initHidden(input_c, device)
        input_length = input_tensor.shape[0]

        for ei in range(input_length):
            encoder_output, encoder_hidden, mu, logvar = self.encoder(input_tensor[ei], encoder_hidden)

        #  z = self.reparameterize(mu, logvar)
        z = mu

        decoder_input = torch.tensor([[SOS_token]], device=device)

        target_c = self.embedding(target_c)
        decoder_hidden = torch.cat((z, target_c), 2)
        decoder_hidden = self.fc(decoder_hidden)

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden= self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words

    def generate(self, z):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = self.fc(z)

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden= self.decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                break
            else:
                decoded_words.append(topi.item())

            decoder_input = topi.squeeze().detach()

        return decoded_words

    def train(self, input_tensor, input_c, criterion):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_c = self.embedding(input_c)
        encoder_hidden = self.encoder.initHidden(input_c, device)

        input_length = input_tensor.shape[0]

        ce_loss = 0
        kl_loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden, mu, logvar = self.encoder(input_tensor[ei], encoder_hidden)

            z = self.reparameterize(mu, logvar)

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = torch.cat((z, input_c), 2)
            decoder_hidden = self.fc(decoder_hidden)


        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False


        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(input_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                _ce_loss, _kl_loss = criterion(decoder_output, input_tensor[di], mu, logvar)
                ce_loss += _ce_loss
                decoder_input = input_tensor[di]  # Teacher forcing
            kl_loss += _kl_loss

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(input_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                _ce_loss, _kl_loss = criterion(decoder_output, input_tensor[di], mu, logvar)
                ce_loss += _ce_loss
                if decoder_input.item() == EOS_token:
                    break
            kl_loss += _kl_loss
        return ce_loss, kl_loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar) 
        eps = torch.randn_like(std)
        return mu + eps * std
