import torch
import numpy as np
from nltk.tokenize import word_tokenize

class SimpleTokenizer(object):
    '''
    Basic tokenizer for caption texts
    '''
    def __init__(self):
        self.word2idx = dict()
        self.idx2word = []
        self.counts = []

        self.add_words = True

        self.add_word('<unk>')
        self.add_word('<sos>')
        self.add_word('<eos>')
        self.counts[:3] = [float('inf')] * 3

        self.UNK = self.word2idx['<unk>']
        self.SOS = self.word2idx['<sos>']
        self.EOS = self.word2idx['<eos>']

    def add_word(self, token):
        idx = len(self.word2idx)
        self.word2idx[token] = idx
        self.idx2word.append(token)
        self.counts.append(0)
        return idx

    def tokenize(self, sentences: str):
        output = []
        for words in sentences:
            tokens = [self.SOS]
            for token in word_tokenize(words):
                idx = self.word2idx.get(token)
                if idx is None:
                    if self.add_words:
                        idx = self.add_word(token)
                    else:
                        idx = self.UNK
                if self.add_words:
                    self.counts[idx] += 1
                tokens.append(idx)
            tokens.append(self.EOS)
            output.append(torch.tensor(tokens))
        return output

    def _get_most_common(self, k):
        if isinstance(self.counts, list):
            self.counts = np.array(self.counts)
        if isinstance(self.idx2word, list):
            self.idx2word = np.array(self.idx2word)
        if k >= len(self.counts):
            return self.idx2word

        top_idx = np.parition(self.counts, kth=k)
        top_words = self.idx2word[top_idx[:k]]
        return top_words

    def decode(self, idxs):
        return ' '.join(self.idx2word[i] for i in idxs)

    def prune(self, max_size):
        pass