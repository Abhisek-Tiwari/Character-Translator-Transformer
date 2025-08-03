import math
import torch
from torch import nn
from torch.nn import functional as F

def scaled_masked_dot_product(Q, K, V, mask=None):
    d_k = Q.size()[-1]
    scaled = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    newVal = torch.matmul(attention, V)
    return newVal, attention

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.QKV_layer = nn.Linear(d_model, 3*d_model)
        self.LinearLayer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size()
        QKV = self.QKV_layer(x)
        QKV = QKV.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
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

class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ffn_hidden, drop_prob=0.1):
        super(PositionwiseFeedForwardNetwork, self).__init__()
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

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForwardNetwork(d_model=d_model, ffn_hidden=ffn_hidden, drop_pron=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, y, mask):
        _y = y
        print("Masked Multihead Attention")
        y = self.self_attention(_y, _y, mask=mask)
        print("Dropout 1")
        y = self.dropout1(y)
        print("Add + Layer Normalization 1")
        y = self.norm1(y+_y)
        _y = y
        print("Masked Encoder Decoder Attention")
        y = self.encoder_decoder_attention(_y, _y, mask=mask)
        print("Dropout 2")
        y = self.dropout2(y)
        print("Add + Layer Normalization 2")
        y = self.norm2(y+_y)
        _y = y
        print("Feed Forward 1")
        y = self.ffn(y)
        print("Dropout 3")
        y = self.dropout3(y)
        print("Add + Layer Normalization 3")
        y = self.norm3(y+_y)
        return y

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers=1):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, mask):
        y = self.layers(x, y)
        return y