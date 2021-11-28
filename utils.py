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
    encoder_dim = 2048

    if args.attention:
        decoder = RNNAttentionDecoder(encoder_dim, args.decoder_hidden_dim, 
            args.decoder_layers, vocab_size, 
            teacher_force_prob=args.teacher_force_prob,
            rnn_cell=rnn_classes[args.decoder_type], key_dim=256)
    else:
        decoder = RNNDecoder(encoder_dim, args.decoder_hidden_dim, 
            args.decoder_layers, vocab_size, 
            teacher_force_prob=args.teacher_force_prob,
            rnn_cell=rnn_classes[args.decoder_type])
    
    return EncoderDecoder(encoder, decoder)

def collate_fn(batch):
    X, Y, Y_attn = [], [], []
    for x, y in batch:
        X.append(x)
        # each time, randomly select a target caption
        idx = random.randint(0, len(y.input_ids)-1)
        Y.append(torch.tensor(y.input_ids[idx]))
        Y_attn.append(torch.tensor(y.attention_mask[idx]))

    X = torch.stack(X, dim=0)
    Y = pad_sequence(Y, batch_first=True)
    Y_attn = pad_sequence(Y_attn, batch_first=True)
    return X, Y, Y_attn