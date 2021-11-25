import os
import argparse
import random
import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.datasets import coco
import torchvision.transforms as transforms
from transformers import T5Tokenizer
from torch.cuda.amp import autocast, GradScaler

from model import *
from tokenizer import SimpleTokenizer
from utils import *

# from pycocoevalcap.eval import COCOEvalCap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, criterion, optimizer, scaler, epoch):
    model.train()
    epoch_loss = 0
    increment = 10
    with tqdm(total=len(loader)) as pbar:
        for i, (x, y, y_attn) in enumerate(loader):
            if (i % increment == 0 and i > 0) or i == len(loader)-1:
                display_loss = epoch_loss / i
                epoch_str = str(epoch+1).rjust(2)
                pbar.set_description(f'[Epoch {epoch_str}] Train Loss: {display_loss:.5f}')
            x = x.to(device)
            y = y.to(device)
            y_attn = y_attn.to(device)
            optimizer.zero_grad()

            with autocast():
                pred = model(x, y)
                vocab_size = pred.shape[-1]
                loss = criterion(pred.view(-1, vocab_size), y.view(-1))
                loss = torch.sum(loss * y_attn.view(-1)) / torch.sum(y_attn)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach()
            pbar.update(1)
    
    return epoch_loss / len(loader)

def validate(model, loader, criterion, epoch, tokenizer):
    model.eval()
    val_loss = 0
    val_bleu = 0
    increment = 10
    preds = []
    with tqdm(total=len(loader)) as pbar:
        for i, (x, y, y_attn) in enumerate(loader):
            if (i % increment == 0 and i > 0) or i == len(loader)-1:
                display_loss = val_loss / i
                epoch_str = str(epoch+1).rjust(2)
                pbar.set_description(f'[Epoch {epoch_str}] Val Loss: {display_loss:.5f}')
            x = x.to(device)
            y = y.to(device)
            y_attn = y_attn.to(device)
            optimizer.zero_grad()

            with autocast():
                pred = model(x, y)
                vocab_size = pred.shape[-1]
                loss = criterion(pred.view(-1, vocab_size), y.view(-1))
                loss = torch.sum(loss * y_attn.view(-1)) / torch.sum(y_attn)
            
            val_loss += loss.detach()
            
            for tokens in pred:
                preds.append(tokenizer.decode(tokens))

            if i == 0:
                print(preds[:5])
    
    return val_loss / len(loader), preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', default='resnet18')
    parser.add_argument('--decoder_type', choices=['rnn', 'lstm', 'gru'], default='lstm')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=10701)
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--pretrained_encoder', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--attention', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default=None)

    args = parser.parse_args()

    if not args.ckpt_dir:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        args.ckpt_dir = os.path.join('checkpoint', datetime.datetime.now().replace(' ', '_'))

    ckpt_file = os.path.join(args.ckpt_dir, 'ckpt.pt')

    # np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    image_transform = transforms.Compose([
        transforms.CenterCrop([360, 640]),
        transforms.ToTensor(),
    ])

    train_data = coco.CocoCaptions(root='data/coco/images/train2014', 
        annFile='data/coco/annotations/captions_train2014.json',
        transform=image_transform, target_transform=tokenizer)

    val_data = coco.CocoCaptions(root='data/coco/images/val2014', 
        annFile='data/coco/annotations/captions_val2014.json',
        transform=image_transform, target_transform=tokenizer)

    vocab_size = tokenizer.vocab_size
    model = make_model(args, vocab_size)
    model.to(device)
    print(model)

    loader_args = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 4 if torch.cuda.is_available() else 0,
        'collate_fn': collate_fn,
        'pin_memory': False,
    }
    train_loader = DataLoader(train_data, **loader_args)

    loader_args['shuffle'] = False
    val_loader = DataLoader(val_data, **loader_args)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    best_loss = float('inf')
    for epoch in args.epochs:
        kwargs = {
            'model': model,
            'loader': train_loader,
            'criterion': criterion,
            'optimizer': optimizer,
            'scaler': scaler,
            'epoch': epoch
        }
        train_loss = train_epoch(**kwargs)
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(kwargs, ckpt_file)
            epoch_str = str(epoch+1).rjust(2)
            print(f'[Epoch {epoch_str}] Saved best model so far with loss {best_loss}')
        
        with torch.no_grad():
            val_loss, preds = validate(model, val_loader, criterion, epoch, tokenizer)
        
        with open('pred.txt', 'w+') as f:
            f.write('\n'.join(preds))
