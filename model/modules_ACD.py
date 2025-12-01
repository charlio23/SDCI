"""
Code extracted from https://github.com/loeweX/AmortizedCausalDiscovery (MIT License).
"""

from torch import nn
import torch.nn.functional as F
import math
import torch
from torch import Tensor

from model import utils

class MLP(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0, use_batch_norm=True, final_linear=False, activation='elu'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.use_batch_norm = use_batch_norm
        self.final_linear = final_linear
        if self.final_linear:
            self.fc_final = nn.Linear(n_out, n_out)

        self.init_weights()
        if activation == 'relu':
            self.activation = F.relu
        if activation == 'softplus':
            self.activation = F.softplus
        else:
            self.activation = F.elu

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        if self.final_linear:
            x = self.fc_final(x)
        if self.use_batch_norm:
            return self.batch_norm(x)
        else:
            return x

class LinAct(nn.Module):
    """A linear layer with a non-linear activation function."""
    def __init__(self, n_in: int, n_out: int, do_prob: float=0, act=None):
        """
        Args:
            n_in: input dimension
            n_out: output dimension
            do_prob: rate of dropout
        """
        super(LinAct, self).__init__()
        if act == None:
            act = nn.ReLU()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_out),
            act,
            nn.Dropout(do_prob)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SelfAtt(nn.Module):
    """Self-attention."""
    def __init__(self, n_in: int, n_out: int):
        """
        Args:
            n_in: input dimension
            n_hid: output dimension
        """
        super(SelfAtt, self).__init__()
        self.query, self.key, self.value = nn.ModuleList([
            nn.Sequential(nn.Linear(n_in, n_out), nn.Tanh())
            for _ in range(3)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., size, dim]

        Return:
            out: [..., size, dim]
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # scaled dot product
        alpha = (query @ key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
        att = alpha.softmax(-1)
        out = att @ value
        return out


class CNN(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.0):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(
            kernel_size=2,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
        )

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = utils.my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


class GNN(nn.Module):
    """
    Reimplementaion of the Message-Passing class in torch-geometric to allow more flexibility.
    """
    def __init__(self):
        super(GNN, self).__init__()
        
    def forward(self, *input):
        raise NotImplementedError

    def propagate(self, x: Tensor, es: Tensor, f_e: Tensor=None, agg: str='mean') -> Tensor:
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            x: [node, ..., dim], node embeddings 
        """
        msg, idx, size = self.message(x, es, f_e)
        x = self.aggregate(msg, idx, size, agg)                          
        return x

    def aggregate(self, msg: Tensor, idx: Tensor, size: int, agg: str='mean') -> Tensor:
        """
        Args:
            msg: [E, ..., dim * 2]
            idx: [E]
            size: number of nodes
            agg: only 3 types of aggregation are supported: 'add', 'mean' or 'max'

        Return:
            aggregated node embeddings
        """
        assert agg in {'add', 'mean', 'max'}
        return scatter(msg, idx, dim_size=size, dim=0, reduce=agg)

    def node2edge(self, x_i: Tensor, x_o: Tensor, f_e: Tensor) -> Tensor:
        """
        Args:
            x_i: [E, ..., dim], embeddings of incoming nodes
            x_o: [E, ..., dim], embeddings of outcoming nodes
            f_e: [E, ..., dim * 2], edge embeddings

        Return:
            edge embeddings
        """
        return torch.cat([x_i, x_o], dim=-1)

    def message(self, x: Tensor, es: Tensor, f_e: Tensor=None, option: str='o2i'):
        """
        Args:
            x: [node, ..., dim], node embeddings
            es: [2, E], edge list
            f_e: [E, ..., dim * 2], edge embeddings
            option: default: 'o2i'
                'o2i': collecting incoming edge embeddings
                'i2o': collecting outcoming edge embeddings

        Return:
            mgs: [E, ..., dim * 2], edge embeddings
            col: [E], indices of 
            size: number of nodes
        """
        if option == 'i2o':
            row, col = es
        if option == 'o2i':
            col, row = es
        else:
            raise ValueError('i2o or o2i')
        x_i, x_o = x[row], x[col]
        msg = self.node2edge(x_i, x_o, f_e)
        return msg, col, len(x)

    def update(self, x):
        return x