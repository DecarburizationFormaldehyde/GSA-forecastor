# -*- coding:utf-8 -*-
# @project: git_project
# @filename: encoder.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/15 9:19
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.sparse_linear import SparseLinear
from utils.utils import attention,clones


class Encoder(nn.Module):
    def __init__(self, encoder_layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(encoder_layer, N)

    def forward(self, x, aux, pos):
        for layer in self.layers:
            x = layer(x, aux, pos)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, gsa_filter, ffd):
        super(EncoderLayer, self).__init__()
        self.gsa_filter = gsa_filter
        self.ffd = ffd

    def forward(self, x, aux, pos):
        x = self.gsa_filter(x, aux, pos)
        return x + self.ffd(x)


def tn_transform_filter(Q, K, M_1, M_2, T):
    tn_trans = torch.zeros((Q.size(0), T, T))
    for i in range(-T + 1, 0 + 1):
        for j in range(-T + 1, 0 + 1):
            l_1 = -min(M_1, T - 1 + min(i, j))
            l_2 = min(M_2, -max(i, j))
            Q_v = Q[:, T - 1 + i + l_1:T - 1 + i + l_2 + 1, :]
            K_v = K[:, T - 1 + j + l_1:T - 1 + j + l_2 + 1, :]
            data = torch.sum(torch.multiply(Q_v, K_v), dim=-1, keepdim=True)
            data = torch.mean(data, dim=1, keepdim=True)
            tn_trans[:, i + T - 1:i + T, j + T - 1:j + T] = data
    return tn_trans


class GSAFilter(nn.Module):
    def __init__(self, h, d_model, d_aux, d_pos, M_1, M_2, T, graph_dependency=None):
        super(GSAFilter, self).__init__()
        assert d_model % h == 0 and d_aux % h == 0 and d_pos % h == 0, \
            "d_model, d_aux, d_pos must be multiple of h"

        self.d_k = d_model // h
        self.d_a = d_aux // h
        self.d_p = d_pos // h

        self.h = h
        self.M_1 = M_1
        self.M_2 = M_2
        self.T = T

        # 线性变换层
        self.nodes_linear = [clones(SparseLinear(self.d_k, d_model, graph_dependency=graph_dependency, reserve=True), h) \
                             for _ in range(3)]
        self.aux_linear = [clones(nn.Linear(d_aux, self.d_a), h) for _ in range(2)]
        self.pos_linear = [clones(nn.Linear(d_pos, self.d_p), h) for _ in range(2)]

        # Wo层
        graph_dependency_repeat = graph_dependency.repeat(h, h)
        self.W_O = SparseLinear(d_model, d_model, graph_dependency=graph_dependency_repeat, reserve=True)

        self.w = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.w_a = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.w_p = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.w_a)
        nn.init.xavier_uniform_(self.w_p)

    def forward(self, x, aux, pos):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        deta = torch.zeros((x.shape[0], x.shape[1], 1)).to(device)
        attn_matrix = torch.zeros((x.shape[0], x.shape[1], 1)).to(device)
        for i in range(self.h):
            Q = F.normalize(self.nodes_linear[0][i](x), p=2, dim=-1)
            K = F.normalize(self.nodes_linear[1][i](x), p=2, dim=-1)
            tn_dot_1 = self.w * tn_transform_filter(Q, K, self.M_1, self.M_2, self.T).to(device)

            QA = F.normalize(self.aux_linear[0][i](aux), p=2, dim=-1)
            KA = F.normalize(self.aux_linear[1][i](aux), p=2, dim=-1)
            dot_2 = self.w_a * torch.matmul(QA, KA.transpose(1, 2))

            QP = F.normalize(self.pos_linear[0][i](pos), p=2, dim=-1)
            KP = F.normalize(self.pos_linear[1][i](pos), p=2, dim=-1)
            dot_3 = self.w_p * torch.matmul(QP, KP.transpose(1, 2))

            all_similarity = tn_dot_1 + dot_2 + dot_3
            attn = attention(all_similarity)
            attn_matrix = torch.cat([attn_matrix, attn], dim=-1)
            score = torch.matmul(attn, self.nodes_linear[2][i](x))
            deta = torch.cat([deta, score], dim=-1)
        return x + self.W_O(deta[:, :, 1:]), (attn_matrix[:, :, 1:], deta[:, :, 1:])

