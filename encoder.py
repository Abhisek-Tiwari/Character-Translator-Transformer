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

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                    for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x