from torch.nn.utils.rnn import pad_sequence
from model import *
import random

rnn_classes = {
    'rnn': nn.RNNCell,
    'lstm': nn.LSTMCell,
    'gru': nn.GRUCell
}

def make_model(args, vocab_size):
    encoder = ResNetEncoder(args.encoder_type, args.pretrained_encoder, 
        not args.attention, args.freeze_encoder)
    encoder_dim = 512

    if args.attention:
        decoder = RNNAttentionDecoder(encoder_dim, args.decoder_layers, vocab_size, 
        rnn_cell=rnn_classes[args.decoder_type], key_dim=256)
    else:
        decoder = RNNDecoder(encoder_dim, args.decoder_layers, vocab_size, 
            rnn_cell=rnn_classes[args.decoder_type])
    
    return EncoderDecoder(encoder, decoder)

def collate_fn(batch):
    X, Y = [], []
    for x, y in batch:
        X.append(x)
        # each time, randomly select a target caption
        Y.append(random.choice(y))
    X = torch.stack(X, dim=0)
    Y = pad_sequence(Y, batch_first=True)
    return X, Y