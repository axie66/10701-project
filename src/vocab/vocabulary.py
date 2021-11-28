import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

class Vocabulary(object):
    def __init__(self,
        vocab_threshold,
        vocab_file="vocab/vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file="../cocoapi/annotations/captions_train2014.json",
        vocab_from_file=True):
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, "rb") as file:
                voc = pickle.load(file)
                self.word2idx = voc.word2idx
                self.idx2word = voc.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            raise ValueError('vocab does not exist')
        
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
