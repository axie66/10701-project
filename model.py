from json import decoder
import torch
from torch import nn
import torch.nn.functional as F
# import clip
from typing import Dict
import pdb
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel

# To fix a bug with downloading resnet weights
torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x=None, y=None, keys=None):
        assert not (x == None and y == None)
        if x is None:
            enc = None
        else:
            enc = self.encoder(x, keys)
        result = self.decoder(enc, y)
        return result

class EncoderTransformerDecoder(EncoderDecoder):
    pass
    # def forward(self, x, y=None, keys=None):
    #     enc = self.encoder(x, keys)
    #     result = self.decoder(enc, y)
    #     return result

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
        model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=pretrained)
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
        self.frozen = freeze
        # if encoder frozen, cache embeddings for efficiency
        self.embed_cache = dict() if freeze else None

    def forward_cache(self, x, key):
        embed = self.embed_cache.get(key)
        if embed is not None:
            return embed
        embed = self.model(x)
        self.embed_cache[key] = embed
        return embed

    def forward(self, x, keys=None):
        if isinstance(x, list):
            embed = [self.forward_cache(img.unsqueeze(0).to(device), k) for img, k in zip(x, keys)]
            return torch.cat(embed, dim=0)
        embed = self.model(x)
        if not self.return_embed:
            # swap channel & spatial dimensions
            embed = embed.transpose(1, 2)
        return embed

# TODO: add weight init
class RNNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, vocab_size, 
                 teacher_force_prob=0.9, rnn_cell=nn.LSTMCell, 
                 rnn_layer_dims=None, dropout_prob=0.25):
        super(RNNDecoder, self).__init__()
        self.num_layers = num_layers
        self.is_lstm = (rnn_cell == nn.LSTMCell)
        # Can change dimensionality

        self.dropout1 = nn.Dropout(dropout_prob)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()

        if rnn_layer_dims is None:
            self.rnn_cells = [rnn_cell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        else:
            self.rnn_cells = [rnn_cell(rnn_layer_dims[i], rnn_layer_dims[i+1])
                              for i in range(len(rnn_layer_dims)-1)]

        for i, cell in enumerate(self.rnn_cells):
            setattr(self, f'rnn_cell{i}', cell)
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.pred_linear = nn.Linear(hidden_dim, vocab_size)
        # Weight tying - could use as improvement to baseline?
        self.embedding.weight = self.pred_linear.weight

        self.teacher_force_prob = teacher_force_prob
        self.max_len = 60
        
        self.init_weights()

    def init_weights(self):
        # Weight init
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if layer is not self.pred_linear:
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.uniform_(layer.weight, -0.1, 0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTMCell):
                nn.init.uniform_(layer.weight_hh, -0.1, 0.1)
                nn.init.uniform_(layer.weight_ih, -0.1, 0.1)

    def forward(self, x, y=None):
        '''
        x: (batch, input_dim) encoded representation of image
        y: (batch, seq_len) ground truth caption output sequence

        Could pre-train as language model before transferring to captioning task
        or just fine-tune GPT

        Experiment with locked dropout
        '''
        hidden_states = [None for _ in range(self.num_layers)]
        if y is not None:
            max_len = y.shape[1]
        else:
            max_len = self.max_len

        batch_size = x.shape[0]
        prediction = torch.zeros(batch_size, 1).to(device)
        # predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(device)
        predictions = []

        x = self.act1(self.input_proj(self.dropout1(x)))

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
            
            prediction = self.pred_linear(self.dropout2(embed))
            # predictions[:, step] = prediction
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
        # query: (batch, key_dim)
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
    def __init__(self, input_dim, hidden_dim, num_layers, vocab_size, teacher_force_prob=0.9, 
                 rnn_cell=nn.LSTMCell, key_dim=256, pretrain=False):
        super(RNNAttentionDecoder, self).__init__(input_dim, hidden_dim, num_layers, 
            vocab_size, teacher_force_prob, rnn_cell, 
            rnn_layer_dims=[hidden_dim + key_dim] + [hidden_dim] * num_layers)
        self.attention = Attention()
        self.key_dim = key_dim
        self.key_transform = nn.Linear(input_dim, key_dim)
        self.value_transform = nn.Linear(input_dim, key_dim)
        self.query_transform = nn.Linear(hidden_dim, key_dim)
        
        del self.dropout1
        del self.input_proj
        del self.act1

        self.hidden_linear = nn.Linear(hidden_dim + key_dim, hidden_dim)

        self.init_weights()
        self.pretrain = pretrain

    def forward(self, x, y=None):
        '''
        x: (batch, img_seq_len, input_dim) encoded representation of image
            to attend to
        y: (batch, seq_len) ground truth caption output sequence
        '''
        hidden_states = [None for _ in range(self.num_layers)]
        if y is not None:
            max_len = y.shape[1]
        else:
            max_len = self.max_len

        if not self.pretrain:
            keys = self.key_transform(x)
            values = self.value_transform(x)

        batch_size = x.shape[0]
        prediction = torch.zeros(batch_size, 1).to(device)
        predictions = []

        batch_size = x.shape[0]

        context = torch.zeros(batch_size, self.key_dim).to(device)

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
            
            if not self.pretrain:
                query = self.query_transform(embed)
                # Note: assume key/value sequences in batch all have same length
                context, _ = self.attention(query, keys, values)

            output = torch.cat([embed, context], dim=1)
            output = self.hidden_linear(output)
            prediction = self.pred_linear(self.dropout2(output))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

class TransformerLMDecoder(nn.Module):
    def __init__(self, model:nn.Module=None, input_dim:int=512, 
                 prefix:torch.LongTensor=None, max_length:int=50):
        super(TransformerLMDecoder, self).__init__()
        self.model = model

        hidden_dim = self.model.config.hidden_size
        mid_dim = (input_dim + hidden_dim) // 2
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        for layer in self.input_proj.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.max_length = max_length
        self.model.config.max_length = max_length
        self.prefix = prefix

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        if self.prefix is not None:
            embeddings = self.model.get_input_embeddings()
            # prefix_embed: (1, prefix_len, embed_dim)
            prefix_embed = embeddings(self.prefix)
            prefix_embed = prefix_embed.expand(x.shape[0], -1, -1)
            x = torch.cat((prefix_embed, x), dim=1)
        return x

    def forward(self, x, y=None):
        raise NotImplementedError

class T5Decoder(TransformerLMDecoder):
    def __init__(self, name, input_dim, prefix=None, max_length=50):
        model = T5ForConditionalGeneration.from_pretrained(name)
        super(T5Decoder, self).__init__(model=model, input_dim=input_dim, 
            prefix=prefix, max_length=max_length)

    def forward(self, x, y=None):
        x = self.encode_input(x)

        if y is None or not self.model.training:
            max_length = self.max_length
            if y is not None:
                max_length = y.shape[1] #+ 1
            # Assume all elements in batch have same sequence length
            attention_mask = torch.ones(x.shape[:2]).to(device)
            # Provide dummy input_ids since we're providing inputs_embeds instead
            # NOTE: requires modification of transformers/generation_utils.py
            # to enable generation from continouous inputs_embeds rather than
            # discrete input_ids
            pred = self.model.generate(
                input_ids=attention_mask, inputs_embeds=x, max_new_tokens=max_length,
                attention_mask=attention_mask, continuous_prompt=True,
                output_scores=True, return_dict_in_generate=True
            )
            # logits, predicted sequences
            return torch.stack(pred.scores, dim=1), pred.sequences

        # Put dummy input 0 at the front
        decoder_input_ids = torch.cat((torch.zeros(y.shape[0], 1).to(y), y[:, :-1]), dim=1)
        labels = y
        out = self.model.forward(
            inputs_embeds=x, decoder_input_ids=decoder_input_ids, labels=labels
        )
        return out.logits, out.loss

class GPT2Decoder(TransformerLMDecoder):
    def __init__(self, input_dim, prefix=None, max_length=50, 
                 sos_token_id=50256, eos_token_id=50256, pad_token_id=-100):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        super(GPT2Decoder, self).__init__(model=model, input_dim=input_dim,
            prefix=prefix, max_length=max_length)
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def forward(self, x, y=None):
        if y is None or not self.model.training:
            max_length = self.max_length
            if y is not None:
                max_length = y.shape[1]
            return self.generate(x, max_length=max_length)
        
        x_embeds = self.encode_input(x)
        # sos = torch.tensor([[self.sos_token_id]]).to(device).expand(x.shape[0], 1)
        # y_filled = torch.cat((sos, torch.where(y==-100, self.eos_token_id, y)), dim=1)
        y_input = y[:, :-1]
        y_filled = torch.where(y_input==-100, self.eos_token_id, y_input)
        y_embeds = self.model.transformer.wte(y_filled)
        inputs_embeds = torch.cat((x_embeds, y_embeds), dim=1)
        out = self.model.forward(inputs_embeds=inputs_embeds, use_cache=False)
        
        # eos = torch.tensor([[self.eos_token_id]]).to(device).expand(x.shape[0], 1)
        # labels = torch.cat((y, eos), dim=1)
        labels = y[:, 1:]
        y_seq_len = labels.shape[1]
        vocab_size = out.logits.shape[-1]
        pred = out.logits[:, -y_seq_len:]

        loss = F.cross_entropy(pred.reshape(-1, vocab_size), labels.reshape(-1), 
            ignore_index=self.pad_token_id)
        # loss = torch.sum(loss * mask.reshape(-1)) / torch.sum(mask)
        return pred, loss

    def generate(self, x, max_length=None):
        # simple greedy search
        if max_length is None:
            max_length = self.max_length

        logits = []
        outputs = self.encode_input(x)
        sos = torch.tensor([[self.sos_token_id]]).to(device).expand(x.shape[0], 1)
        sos_embeds = self.model.transformer.wte(sos)
        outputs = torch.cat((outputs, sos_embeds), dim=1)
        past_key_values = None
        done = torch.zeros(x.shape[0]).to(device)
        tokens = []
        for i in range(max_length):
            if i == 0:
                next_out = self.model(inputs_embeds=outputs, use_cache=True)
            else:
                next_out = self.model(inputs_embeds=next_token_embeds, 
                    past_key_values=past_key_values, use_cache=True)
                
            next_logits = next_out.logits[:, -1].unsqueeze(1)
            logits.append(next_logits)
            
            next_tokens = next_logits.argmax(dim=-1)
            tokens.append(next_tokens)
            next_token_embeds = self.model.transformer.wte(next_tokens)

            # outputs = torch.cat((outputs, next_token_embeds), dim=1)
            past_key_values = next_out.past_key_values

            done += (next_tokens.view(-1) == self.eos_token_id)
            if done.all():
                break

        return torch.cat(logits, dim=1), torch.cat(tokens, dim=1)


if __name__ == '__main__':
    # # Sanity check
    # encoder = ResNetEncoder('resnet18', return_embed=True)
    # decoder = RNNDecoder(512, 2, 10000, rnn_cell=nn.GRUCell)
    # model = EncoderDecoder(encoder, decoder)

    # # batch size 5, 3 input channels, 256x256 image
    x = torch.rand(5, 3, 256, 256)
    # y_pred = model(x)
    # # enc = encoder(x)

    attn_encoder = ResNetEncoder('resnet18', return_embed=False)
    attn_decoder = RNNAttentionDecoder(512, 256, 2, 10000, rnn_cell=nn.GRUCell)
    attn_model = EncoderDecoder(attn_encoder, attn_decoder)
    y2 = attn_model(x)

    # model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    #     'google/vit-base-patch16-224-in21k', 'openai-gpt'
    # )

    # t5 = T5Decoder.from_pretrained('t5-small')
    # model = EncoderDecoder(attn_encoder, t5)
