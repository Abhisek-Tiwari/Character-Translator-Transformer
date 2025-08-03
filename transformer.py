import math
import torch
from torch import nn
from torch.nn import functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_masked_dot_product(Q, K, V, mask=None):
    d_k = Q.size()[-1]
    scaled = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    newVal = torch.matmul(attention, V)
    return newVal, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.QKV_layer = nn.Linear(d_model, 3*d_model)
        self.LinearLayer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        QKV = self.QKV_layer(x)
        QKV = QKV.reshape(batch_size, sequence_length, self.num_heads, 3*self.head_dim)
        QKV = QKV.permute(0, 2, 1, 3)
        Q, K, V = QKV.chunk(3, dim=-1)
        values, attentions = scaled_masked_dot_product(Q, K, V, mask)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.LinearLayer(values)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.Q_layer = nn.Linear(d_model, d_model)
        self.KV_layer = nn.Linear(d_model, 2*d_model)
        self.LinearLayer = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):
        batch_size, sequence_length, d_model = x.size()
        KV = self.KV_layer(x)
        KV = KV.reshape(batch_size, sequence_length, self.num_heads, 2*self.head_dim)
        KV = KV.permute(0, 2, 1, 3)
        K, V = KV.chunk(2, dim=-1)
        Q = self.Q_layer(y)
        Q = Q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)
        values, attentions = scaled_masked_dot_product(Q, K, V, mask)
        values = values.reshape(batch_size, sequence_length, d_model)
        out = self.LinearLayer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, epsilon=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, input):
        dims = [-(i+1) for i in range(len(self.parameters_shape))]
        mean = input.mean(dims, keepdim = True)
        var = ((input-mean)**2).mean(dim=dims, keepdim = True)
        std = (var+self.epsilon).sqrt()
        y = (input-mean)/std
        out = self.gamma*y + self.beta
        return out

class PositionalFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
        super(PositionalFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class SentenceEmbedding(nn.Module):
    def __init__(self, d_model, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

    def batch_tokenize(self, batch, start_token=True, end_token=True):

        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indicis = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indicis.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicis.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicis), self.max_sequence_length):
                sentence_word_indicis.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicis)

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())

    def forward(self, x, end_token=True):  # sentence
        x = self.batch_tokenize(x, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads= num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionalFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residual = x
        x = self.attention(x, mask=None)
        x = self.dropout1(x)
        x = self.norm1(x+residual)
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+residual)
        return x


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attn_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attn_mask)
        return x

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y

class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embeddings = SentenceEmbedding(d_model, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                    for _ in range(num_layers)])

    def forward(self, x, self_attn_mask, start_token, end_token):
        x = self.sentence_embeddings(x, start_token, end_token)
        x = self.layers(x, self_attn_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionalFeedForward(d_model=d_model, ffn_hidden=ffn_hidden, drop_pron=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, y, mask):
        _y = y
        y = self.self_attention(_y, _y, mask=mask)
        y = self.dropout1(y)
        y = self.norm1(y+_y)
        _y = y
        y = self.encoder_decoder_attention(_y, _y, mask=mask)
        y = self.dropout2(y)
        y = self.norm2(y+_y)
        _y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y+_y)
        return y


class Decoder(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embeddings = SentenceEmbedding(d_model, max_sequence_length, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attn_mask, cross_attn_mask, start_token, end_token):
        y = self.sentence_embeddings(y, start_token, end_token)
        y = self.layers(x, y, self_attn_mask, cross_attn_mask)
        return y


class Transformer(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 num_heads,
                 drop_prob,
                 num_layers,
                 max_sequence_length,
                 kn_vocab_size,
                 kannada_to_index,
                 english_to_index,
                 START_TOKEN,
                 END_TOKEN,
                 PADDING_TOKEN
                 ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN ,END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self,
                x,
                y,
                encoder_self_attn_mask=None,
                decoder_self_attn_mask=None,
                decoder_cross_attn_mask=None,
                encoder_start_token=False,
                encoder_end_token=False,
                decoder_start_token=False,
                decoder_end_token=False):
        x = self.encoder(x, encoder_self_attn_mask, start_token=encoder_start_token, end_token=encoder_end_token)
        output = self.decoder(x, y, decoder_self_attn_mask, decoder_cross_attn_mask, start_token=decoder_start_token, end_token=decoder_end_token)
        output = self.linear(output)
        return output