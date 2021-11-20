import argparse
import pickle
import random

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import coco
import torchvision.transforms as transforms

from model import *
from tokenizer import SimpleTokenizer
from utils import *

def train_epoch(model, train_loader, args, epoch):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', default='resnet18')
    parser.add_argument('--decoder_type', choices=['rnn', 'lstm', 'gru'], default='lstm')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=10701)
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--pretrained_encoder', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--attention', action='store_true')

    args = parser.parse_args()

    # np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # tokenizer = SimpleTokenizer()
    with open('tokenizer.p', 'rb') as f:
        tokenizer = pickle.load(f)
    tokenizer.add_words = False

    image_transform = transforms.Compose([
        transforms.CenterCrop([360, 640]),
        transforms.ToTensor(),
    ])

    train_data = coco.CocoCaptions(root='data/coco/images/train2014', 
        annFile='data/coco/annotations/captions_train2014.json',
        transform=image_transform, target_transform=tokenizer.tokenize)

    # val_data = coco.CocoCaptions(root='data/coco/images/val2014', 
    #     annFile='data/coco/annotations/captions_val2014.json',
    #     transform=transforms.ToTensor(), target_transform=tokenizer.tokenize)

    vocab_size = len(tokenizer.word2idx)
    model = make_model(args, vocab_size)

    loader_args = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 4 if torch.cuda.is_available() else 0,
        'collate_fn': collate_fn,
        'pin_memory': False,
    }
    train_loader = DataLoader(train_data, **loader_args)

    x, y = next(iter(train_loader))

    # TODO: training loop, evaluation, etc