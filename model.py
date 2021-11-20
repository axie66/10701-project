import torch
from torch import nn
# import clip
from typing import Dict
import pdb
from transformers import EncoderDecoderModel

# To fix a bug with downloading resnet18 weights
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y=None):
        enc = self.encoder(x)
        result = self.decoder(enc, y)
        return result

def init_weight(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)

class ResNetEncoder(nn.Module):
    '''
    Use pre-trained ResNet as a feature extractor
    '''
    def __init__(self, name, pretrained=True, return_embed=True, freeze=True):
        '''
        name: specific ResNet version used
        pretrained: whether to use pretrained ResNet weights
        return_embed: if true, return fixed-length embedding of input image,
            otherwise return final layer feature map for visual attention
        '''
        super(ResNetEncoder, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 
                                    name, pretrained=pretrained)
        self.return_embed = return_embed
        if self.return_embed:
            self.model = nn.Sequential(
                *list(model.children())[:-1], 
                nn.Flatten() # Flatten all the 1x1 feature maps
            )
        else:
            self.model = nn.Sequential(
                *list(model.children())[:-2], 
                nn.Flatten(start_dim=2) # Flatten 
            )
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        # if encoder frozen, cache embeddings for efficiency
        self.embed_cache = dict() if freeze else None

    def forward(self, x, key=None):
        if self.return_embed and self.embed_cache is not None and key is not None:
            embed = self.embed_cache.get(key)
            if embed is not None:
                return embed
        embed = self.model(x)
        if not self.return_embed:
            # swap channel & spatial dimensions
            embed = embed.transpose(1, 2)
        elif self.embed_cache is not None and key is not None:
            self.embed_cache[key] = embed
        return embed

# TODO: add weight init
class RNNDecoder(nn.Module):
    def __init__(self, input_dim, num_layers, vocab_size, 
                 teacher_force_prob=0.9, rnn_cell=nn.LSTMCell):
        super(RNNDecoder, self).__init__()
        self.num_layers = num_layers
        self.is_lstm = (rnn_cell == nn.LSTMCell)
        # Can change dimensionality
        self.rnn_cells = [rnn_cell(input_dim, input_dim) for _ in range(num_layers)]
        
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.pred_linear = nn.Linear(input_dim, vocab_size)
        # Weight tying - could use as improvement to baseline?
        self.embedding.weight = self.pred_linear.weight

        self.teacher_force_prob = teacher_force_prob
        self.max_len = 300

    def forward(self, x, y=None):
        '''
        x: (batch, input_dim) encoded representation of image
        y: (batch, seq_len) ground truth caption output sequence

        Could pre-train as language model before transferring to captioning task
        or just fine-tune GPT

        Experiment with locked dropout
        '''
        hidden_states = [None for _ in range(self.num_layers)]
        if self.training and y is not None:
            max_len = y.shape[1]
        else:
            max_len = self.max_len

        batch_size = x.shape[0]
        prediction = torch.zeros(batch_size, 1).to(device)
        predictions = []

        for step in range(max_len):
            if step == 0:
                embed = x
            elif self.training and torch.rand(1) < self.teacher_force_prob and y is not None:
                embed = self.embedding(y[:, step-1])
            else:
                embed = self.embedding(prediction.argmax(dim=-1))

            for i, cell in enumerate(self.rnn_cells):
                hidden_states[i] = cell(embed, hidden_states[i])
                embed = hidden_states[i][0] if self.is_lstm else hidden_states[i]
            
            prediction = self.pred_linear(embed)
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

class Attention(nn.Module):
    '''
    Scaled dot-product attention
    TODO: multi-head attention
    '''
    BIG_NUM = 1e9
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # query: (batch, )
        key_value_size = query.shape[1]
        energy = torch.bmm(key, query.unsqueeze(2))
        energy = energy.squeeze(2) / key_value_size**0.5
        if mask is not None:
            energy = energy + (mask.float()-1) * Attention.BIG_NUM
        attn = self.softmax(energy)
        context = torch.bmm(attn.unsqueeze(1), value)
        context = context.squeeze(1)
        return context, attn

class RNNAttentionDecoder(RNNDecoder):
    def __init__(self, input_dim, num_layers, vocab_size, teacher_force_prob=0.9, 
                 rnn_cell=nn.LSTMCell, key_dim=256):
        super(RNNAttentionDecoder, self).__init__(input_dim, num_layers, 
            vocab_size, teacher_force_prob, rnn_cell)
        
        self.attention = Attention()
        self.key_dim = key_dim
        self.key_transform = nn.Linear(input_dim, key_dim)
        self.value_transform = nn.Linear(input_dim, key_dim)
        self.query_transform = nn.Linear(input_dim, key_dim)

        self.pred_linear = nn.Linear(input_dim + key_dim, vocab_size)

    def forward(self, x, y=None):
        '''
        x: (batch, img_seq_len, input_dim) encoded representation of image
            to attend to
        y: (batch, seq_len) ground truth caption output sequence
        '''
        hidden_states = [None for _ in range(self.num_layers)]
        if self.training and y is not None:
            max_len = y.shape[1]
        else:
            max_len = self.max_len

        keys = self.key_transform(x)
        values = self.value_transform(x)

        batch_size = x.shape[0]
        prediction = torch.zeros(batch_size, 1).to(device)
        predictions = []

        batch_size = x.shape[0]

        context = torch.zeros(batch_size, self.key_dim)

        for step in range(max_len):
            if self.training and torch.rand(1) < self.teacher_force_prob and y is not None and step > 0:
                # Teacher forcing
                embed = self.embedding(y[:, step-1])
            else:
                embed = self.embedding(prediction.argmax(dim=-1))

            embed = torch.cat([embed, context], dim=1)

            for i, cell in enumerate(self.rnn_cells):
                hidden_states[i] = cell(embed, hidden_states[i])
                embed = hidden_states[i][0] if self.is_lstm else hidden_states[i]
            
            query = self.query_transform(embed)
            # Note: assume key/value sequences in batch all have same length
            context, _ = self.attention(query, keys, values)

            output = torch.cat([embed, context], dim=1)

            prediction = self.pred_linear(output)
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

class MyTransformer(EncoderDecoderModel):
    def forward(self, input_ids, decoder_ids, attention_mask):
        pass

if __name__ == '__main__':
    # Sanity check
    encoder = ResNetEncoder('resnet18', return_embed=True)
    decoder = RNNDecoder(512, 2, 10000, rnn_cell=nn.GRUCell)
    model = EncoderDecoder(encoder, decoder)

    # batch size 5, 3 input channels, 256x256 image
    x = torch.rand(5, 3, 256, 256)
    y_pred = model(x)
    # enc = encoder(x)

    attn_encoder = ResNetEncoder('resnet18', return_embed=False)
    attn_decoder = RNNAttentionDecoder(512, 2, 10000, rnn_cell=nn.GRUCell)
    attn_model = EncoderDecoder(encoder, decoder)
    y2 = attn_model(x)