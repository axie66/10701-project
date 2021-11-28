import os
import argparse
import random
import datetime
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import (
    T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer,
    AdamW, get_linear_schedule_with_warmup
)
from transformers.optimization import Adafactor

from data import CocoCaptions
from model import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    epoch_loss = 0
    increment = 10
    with tqdm(total=len(loader)) as pbar:
        for i, (x, y, y_attn, img_ids) in enumerate(loader):
            if (i % increment == 0 and i > 0) or i == len(loader)-1:
                display_loss = epoch_loss / i
                epoch_str = str(epoch+1).rjust(2)
                pbar.set_description(f'[Epoch {epoch_str}] Train Loss: {display_loss:.5f}')
            x = x.to(device) if not isinstance(x, list) else x
            y = y.to(device)
            # y_attn = y_attn.to(device)
            optimizer.zero_grad()

            _, loss = model(x, y, keys=img_ids)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.detach()
            pbar.update(1)
    
    return epoch_loss / len(loader)

def validate(model, loader, epoch, tokenizer):
    model.eval()
    val_loss = 0
    increment = 10
    preds = []
    with tqdm(total=len(loader)) as pbar:
        for i, (x, y, y_attn, img_ids) in enumerate(loader):
            if (i % increment == 0 and i > 0) or i == len(loader)-1:
                display_loss = val_loss / i
                epoch_str = str(epoch+1).rjust(2)
                pbar.set_description(f'[Epoch {epoch_str}] Val Loss: {display_loss:.5f}')
            x = x.to(device) if not isinstance(x, list) else x
            y = y.to(device)
            # y_attn = y_attn.to(device)

            # scores: (batch_size, seq_len)
            # pred_tokens: (batch_size, seq_len+1)
            y = y[:, 1:] # for gpt2 only
            scores, pred_tokens = model(x, y, keys=img_ids)

            batch_size, pred_seq_len, vocab_dim = scores.shape
            actual_seq_len = y.shape[1]
            if pred_seq_len < actual_seq_len:
                new_scores = torch.zeros(batch_size, actual_seq_len, vocab_dim, device=device)
                new_scores[:, :pred_seq_len] = scores
                scores = new_scores
            loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), y.reshape(-1),
                ignore_index=model.decoder.pad_token_id)
            
            val_loss += loss.detach()
            
            for img_id, tokens in zip(img_ids, pred_tokens):
                decoded = tokenizer.decode(tokens)
                eos_token = '<|endoftext|>' # for GPT2
                # eos_token = '</s>' # for T5
                eos_idx = decoded.find(eos_token)
                if eos_idx != -1:
                    decoded = decoded[:eos_idx]
                preds.append({'image_id': int(img_id), 'caption': decoded.strip()})
                # preds[int(img_id)] = decoded[:eos_idx+len(eos_token)]
            
            if i == 0:
                print(preds[:5])

            pbar.update(1)
    
    return val_loss / len(loader), preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_type', default='resnet18')
    parser.add_argument('--decoder_type', choices=['t5', 'gpt2'], default='t5')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=10701)
    parser.add_argument('--pretrained_encoder', action='store_true')
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--freeze_decoder', action='store_true')
    parser.add_argument('--adafactor', action='store_true')
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--first_epoch_warmup', type=float, default=0.25)

    args = parser.parse_args()

    from_checkpoint = (args.ckpt_dir is not None)

    if not from_checkpoint: # new run
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        args.ckpt_dir = os.path.join('checkpoint', str(datetime.datetime.now()).replace(' ', '_'))
        if not os.path.isdir(args.ckpt_dir):
            os.mkdir(args.ckpt_dir)
        start_epoch = 0
    else: # starting from previous checkpoint
        assert os.path.isdir(args.ckpt_dir)
        ckpt_file = os.path.join(args.ckpt_dir, 'ckpt.pt')
        state = torch.load(ckpt_file)
        model = state['model']
        optimizer = state['optimizer']
        start_epoch = state['epoch'] + 1


    print(args)
    print()
    print(f"Saving to directory {args.ckpt_dir}.\n")

    ckpt_file = os.path.join(args.ckpt_dir, 'ckpt.pt')
    args_file = os.path.join(args.ckpt_dir, 'args.txt')
    with open(args_file, 'w+') as f:
        f.write(str(args))

    # np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.decoder_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        target_transform = tokenizer
        collate_fn = CocoCaptions.collate_fn
    elif args.decoder_type == 'gpt2':
        pad_token_id = -100
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        target_transform = tokenizer
        collate_fn = lambda b: CocoCaptions.collate_fn(b, 
            tokenizer.bos_token_id, tokenizer.eos_token_id, pad_token_id)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_data = CocoCaptions(root='data/coco/images/train2014', 
        annFile='data/coco/annotations/captions_train2014.json',
        transform=image_transform, target_transform=target_transform)

    val_data = CocoCaptions(root='data/coco/images/val2014', 
        annFile='data/coco/annotations/captions_val2014.json',
        transform=image_transform, target_transform=target_transform)

    if not from_checkpoint:
        vocab_size = tokenizer.vocab_size
        encoder = ResNetEncoder(args.encoder_type, args.pretrained_encoder, 
            return_embed=False, freeze=args.freeze_encoder)
        encoder_dim = 2048 if args.encoder_type == 'resnet50' else 512
        
        if args.decoder_type == 't5':
            prefix = torch.tensor(tokenizer(['caption:']).input_ids)[:, :-1].to(device)
            decoder = T5Decoder('t5-small', encoder_dim, prefix=prefix)
        else:
            decoder = GPT2Decoder(encoder_dim, sos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=pad_token_id)
        
        model = EncoderTransformerDecoder(encoder, decoder)
        model.to(device)

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

    if args.freeze_decoder:
        if not from_checkpoint:
            optimizer = AdamW(model.decoder.input_proj.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(len(train_loader) * args.first_epoch_warmup), 
            num_training_steps=(args.epochs - start_epoch) * len(train_loader)
        )
    elif args.adafactor:
        # optimizer = AdamW(model.parameters(), lr=args.lr)
        # Adafactor lr should be relatively large, around 1e-3
        if not from_checkpoint:
            optimizer = Adafactor(model.parameters(), scale_parameter=False, 
                relative_step=False, warmup_init=False, lr=args.lr)
        scheduler = None
    else:
        if not from_checkpoint:
            optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(len(train_loader) * args.first_epoch_warmup), 
            num_training_steps=(args.epochs - start_epoch) * len(train_loader)
        )

    if from_checkpoint:
        with torch.no_grad():
            val_loss, preds = validate(model, val_loader, 0, tokenizer)
        torch.cuda.empty_cache()
        pred_file = os.path.join(args.ckpt_dir, f'pred0.json')
        with open(pred_file, 'w+') as f:
            json.dump(preds, f)

    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs):
        kwargs = {
            'model': model,
            'loader': train_loader,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': epoch
        }
        train_loss = train_epoch(**kwargs)
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            val_loss, preds = validate(model, val_loader, epoch, tokenizer)
        torch.cuda.empty_cache()
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model': model,
                'optimizer': optimizer,
                'epoch': epoch
            }, ckpt_file)
            epoch_str = str(epoch+1).rjust(2)
            print(f'[Epoch {epoch_str}] Saved best model so far with loss {best_loss}')

        pred_file = os.path.join(args.ckpt_dir, f'pred{epoch+1}.json')
        with open(pred_file, 'w+') as f:
            json.dump(preds, f)
