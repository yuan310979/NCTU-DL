import re
import time
import string
import random
import itertools

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.time import timeSince
from utils.data.dataloader import prepareData, readLangs
from nn.module.encoder import EncoderRNN 
from nn.module.decoder import DecoderRNN
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from model import CVAE

idx2tense = {0: 'sp', 1:'tp', 2:'pg', 3:'p'}
#  tense2oh = {'sp': [1, 0, 0, 0, 1, 0, 0, 0], 
         #  'tp': [0, 1, 0, 0, 0, 1, 0, 0],
         #  'pg': [0, 0, 1, 0, 0, 0, 1, 0],
         #  'p': [0, 0 , 0, 1, 0, 0 , 0, 1]}

tense2index = {'sp': 0, 
         'tp': 1,
         'pg': 2,
         'p': 3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

input_lang, pairs = prepareData('lang', "./data/train.txt")
torch.save(input_lang, "lang_class.pth")

print(random.choice(pairs))

def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in word]

def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorFromTense(tense):
    input_c = tense2index[tense]
    input_c = torch.tensor(input_c, dtype=torch.long, device=device).view(1, -1)
    return input_c

#  def tensorFromIdx(idx):
    #  input_c = tense2oh[idx2tense[idx]]
    #  input_c = torch.tensor(input_c, dtype=torch.float, device=device).view(1, 1, -1)
    #  return input_c

def tensorsFromPair(pair, random=True):
    if random == True:
        idx = np.random.randint(0, 4, 1)[0]
        tense = idx2tense[idx]
        #  input_c = torch.tensor(tense2oh[tense], dtype=torch.long, device=device).view(-1, 1)
        input_c = torch.tensor(tense2index[tense], dtype=torch.long, device=device).view(-1, 1)
        input_tensor = tensorFromWord(input_lang, pair[idx])

    return (input_tensor, input_tensor), (input_c, input_c)

def loss_func(x, y, mu, logvar):
    CE = F.cross_entropy(x, y)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return CE, KLD 

def bleu_score(a, b):
    cc = SmoothingFunction()
    bl = sentence_bleu([a], b, smoothing_function=cc.method1)
    return bl

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar) 
    eps = torch.randn_like(std)
    return mu + eps * std

teacher_forcing_ratio = 0.5 

def train(input_tensor, target_tensor, input_c, target_c, model, optimizer, criterion, weight, max_length=MAX_LENGTH):
    optimizer.zero_grad()

    input_c = input_c.long().view(1, -1)
    input_length = input_tensor.shape[0]

    ce_loss, kl_loss = model.train(input_tensor, input_c, loss_func) 
    loss = ce_loss + kl_loss * weight
    loss.backward()

    optimizer.step()

    return loss.item() / input_length, ce_loss.item() / input_length, kl_loss / input_length

def kl_annealing(i):
    weight = 0.0
    if i > 30000:
        i = i % 5000
        weight = 1 / 5000 * i
    return weight
        

def trainIters(model, n_iters, print_every=1000, plot_every=1, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    plot_kl_losses = []
    plot_ce_losses = []
    plot_bleu_score = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    best_bleu_score = 0

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    training_pairs = []
    training_cs = []

    for i in range(n_iters):
        pair = random.choice(pairs)
        tfp = tensorsFromPair(pair) 
        training_pairs.append(tfp[0])
        training_cs.append(tfp[1])

    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        training_c = training_cs[iter - 1]

        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        input_c = training_c[0]
        target_c = training_c[1]

        loss, ce_loss, kl_loss = train(input_tensor, target_tensor, input_c, target_c, model, optimizer, criterion, kl_annealing(iter))
        print_loss_total += loss
        plot_loss_total += loss

        bleu = evaluateByTestData(model)
        if best_bleu_score < bleu:
            best_bleu_score = bleu

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            print("BestBleu: {:.4f}".format(best_bleu_score))
            print("Bleu: {:.4f}".format(bleu))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_kl_losses.append(kl_loss.item())
            plot_ce_losses.append(ce_loss)
            plot_bleu_score.append(bleu)
            plot_loss_total = 0

        if bleu > 0.7:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'bleu': bleu,
                'ce_loss': ce_loss,
                'kl_loss': kl_loss
            }, f'./checkpoint/{bleu}_{iter}.pth')

    showPlot(plot_losses)

    torch.save({
        'kl_loss': plot_kl_losses,
        'ce_loss': plot_ce_losses,
        'loss': plot_losses,
        'bleu_score': plot_bleu_score 
    }, "plot_result2")



######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def evaluate(model, c_pair, word, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromWord(input_lang, word)
        input_c = tensorFromTense(c_pair[0])
        target_c = tensorFromTense(c_pair[1])

        decoded_idxs = model(input_tensor, input_c, target_c)
        decoded_words = ''.join(input_lang.index2char[i] for i in decoded_idxs)

        return decoded_words

def evaluateByTestData(model):
    _, pairs = readLangs("test_p", "./data/test.txt")
    _, c_pairs = readLangs("test_c", "./data/test_c.txt")

    total_bleu_score = 0.0
    for pair, c_pair in zip(pairs, c_pairs):
        #  print('>', pair[0])
        #  print('=', pair[1])
        output_words = evaluate(model, c_pair, pair[0])
        output_sentence = ''.join(output_words)
        #  print('<', output_sentence)
        #  print('')
        total_bleu_score += bleu_score(pair[1], output_sentence)
        _bleu_score = total_bleu_score / len(pairs)
    #  print(f"Bleu Score: {_bleu_score}")
    return _bleu_score
        
def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        idx = np.random.randint(0, 4)
        print('>', pair[idx])
        print('=', pair[idx])
        output_words = evaluate(encoder, decoder, idx, pair[idx])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def save_checkpoint(state, filename):
    torch.save(state, filename)


hidden_size = 256
latent_size = 32 
model = CVAE(input_lang.n_words, hidden_size, latent_size, input_lang.n_words).to(device)

trainIters(model, 100000, print_every=5000)

evaluateByTestData(model)
