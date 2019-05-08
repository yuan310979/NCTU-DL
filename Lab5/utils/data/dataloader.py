import re
import unicodedata

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addWord(self, word):
        for char in word:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_words
            self.char2count[char] = 1
            self.index2char[self.n_words] = char 
            self.n_words += 1
        else:
            self.char2count[char] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang, path):
    # Read the file and split into lines
    lines = open(path, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split(' ')] for l in lines]

    input_lang = Lang(lang)

    return input_lang, pairs

def prepareData(lang, path):
    input_lang, pairs = readLangs(lang, path)
    print("Read %s words sequences" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        for p in pair:
            input_lang.addWord(p)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    return input_lang, pairs
