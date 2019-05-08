import torch

from model import CVAE
from utils.data.dataloader import prepareData, readLangs
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

MAX_LENGTH = 20
EOS_token = 1

def bleu_score(a, b):
    cc = SmoothingFunction()
    bl = sentence_bleu([a], b, smoothing_function=cc.method1)
    return bl

def tensorFromTense(tense):
    input_c = tense2index[tense]
    input_c = torch.tensor(input_c, dtype=torch.long, device=device).view(1, -1)
    return input_c

def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in word]

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
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(model, c_pair, pair[0])
        output_sentence = ''.join(output_words)
        print('<', output_sentence)
        print('')
        total_bleu_score += bleu_score(pair[1], output_sentence)
        _bleu_score = total_bleu_score / len(pairs)
    print(f"Bleu Score: {_bleu_score}")
    return _bleu_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tense2index = {'sp': 0, 'tp': 1,'pg': 2,'p': 3}

latent_size = 32
hidden_size = 256

input_lang = torch.load("./lang_class.pth")

checkpoint = torch.load("./checkpoint/0.7902374299152759_61208.pth")
model = CVAE(28, hidden_size, latent_size, 28).to(device)
model.load_state_dict(checkpoint['state_dict'])

evaluateByTestData(model)
