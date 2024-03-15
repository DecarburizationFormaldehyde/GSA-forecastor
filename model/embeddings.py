# -*- coding:utf-8 -*-
# @project: git_project
# @filename: embeddings.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/15 9:46
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.sparse_linear import SparseLinear


class GraphEmbeddings(nn.Module):
    def __init__(self, nodes_size: int, d_model: int, graph_dependency=None):
        """
        a feedforward neural network  implemented with sparse linear
        layers and nonlinear activation functions
        """
        super(GraphEmbeddings, self).__init__()
        self.ffn = SparseLinear(nodes_size, d_model, graph_dependency=graph_dependency)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ffn(x))


class AuxEmbeddings(nn.Module):
    def __init__(self, a_dim: int, d_aux: int):
        """
        a feedforward neural network  implemented with full-linear
        layers and nonlinear activation functions
        """
        super(AuxEmbeddings, self).__init__()
        self.ffn = nn.Linear(a_dim, d_aux)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.ffn(x))


class PosEmbeddings(nn.Module):
    "Implement the PE function."
    def __init__(self, d_pos, dropout, max_len=5000):
        super(PosEmbeddings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_pos)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_pos, 2) *
                             -(math.log(10000.0) / d_pos))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = Variable(self.pe[:, :x.size(1)],requires_grad=True)
        return self.dropout(x)