import torch
import torch.nn as nn

from model.sparse_linear import SparseLinear
from utils.utils import attention,clones
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, decoder_layer, N, de_embed):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, N)
        self.de_embed = de_embed

    def forward(self, k, memory, x_pre, aux, pos):
        for layer in self.layers:
            x_pre = layer(k, memory, x_pre, aux, pos)
        return self.de_embed(x_pre)


class DecoderLayer(nn.Module):
    def __init__(self, gsa_pre, ffd):
        super(DecoderLayer, self).__init__()
        self.gsa_pre = gsa_pre
        self.ffd = ffd

    def forward(self, k, memory, x_pre, aux, pos):
        x = self.gsa_pre(k, memory, x_pre, aux, pos)
        return x + self.ffd(x)


def tn_transform_pre(Q, K, M, T, k):
    Q_v = Q[:, Q.size(1) - M:, :]
    tn_trans = torch.zeros((Q.size(0), 1, k + T - M + 1))
    for i in range(-T + M, k+1):
        if T + i - (T - M + i) != M:
            dd=i
            aa=Q_v.shape
            bb=K_v.shape
        K_v = K[:, T - M + i:T + i, :]
        data = torch.sum(torch.multiply(Q_v, K_v), dim=-1, keepdim=True)
        data = torch.mean(data, dim=1, keepdim=True)
        tn_trans[:, 0:1, T - M + i:i + T - M + 1] = data
    return tn_trans


class GSAPredict(nn.Module):
    def __init__(self, h, d_model, d_aux, d_pos, M, T, gru, graph_dependency=None):
        super(GSAPredict, self).__init__()
        assert d_model % h == 0 and d_aux % h == 0 and d_pos % h == 0, \
            "d_model, d_aux, d_pos must be multiple of h"

        self.d_k = d_model // h
        self.d_a = d_aux // h
        self.d_p = d_pos // h

        self.h = h
        self.M = M
        self.T = T

        # 线性变换层
        self.nodes_linear = [clones(SparseLinear(self.d_k, d_model, graph_dependency=graph_dependency, reserve=True), h) \
                             for _ in range(3)]
        self.aux_linear = [clones(nn.Linear(d_aux, self.d_a), h) for _ in range(2)]
        self.pos_linear = [clones(nn.Linear(d_pos, self.d_p), h) for _ in range(2)]

        # Wo层
        graph_dependency_repeat = graph_dependency.repeat(h, h)
        self.W_O = SparseLinear(d_model, d_model, graph_dependency=graph_dependency_repeat, reserve=True)

        self.gru = gru
        self.w = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.w_a = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.w_p = nn.Parameter(torch.FloatTensor(1, 1), requires_grad=True)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.w_a)
        nn.init.xavier_uniform_(self.w_p)

    def forward(self, k, memory, x_pre, aux, pos):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        deta = torch.zeros((x_pre.shape[0], x_pre.shape[1], 1))
        x = torch.cat([memory, x_pre], dim=1)
        for i in range(self.h):
            Q = F.normalize(self.nodes_linear[0][i](x), p=2, dim=-1)
            K = F.normalize(self.nodes_linear[1][i](x), p=2, dim=-1)
            tn_dot_1 = self.w * tn_transform_pre(Q, K, self.M, self.T, k).to(device)

            QA = F.normalize(self.aux_linear[0][i](aux), p=2, dim=-1)
            KA = F.normalize(self.aux_linear[1][i](aux), p=2, dim=-1)
            dot_2 = self.w_a * torch.matmul(QA, KA.transpose(1, 2))[:, -1:, KA.size(1) - k - self.T + self.M - 1:]

            QP = F.normalize(self.pos_linear[0][i](pos), p=2, dim=-1)
            KP = F.normalize(self.pos_linear[1][i](pos), p=2, dim=-1)
            dot_3 = self.w_p * torch.matmul(QP, KP.transpose(1, 2))[:, -1:, KP.size(1) - k - self.T + self.M - 1:]

            all_similarity = tn_dot_1 + dot_2 + dot_3
            attn = attention(all_similarity)

            history_deta = torch.matmul(attn[:, :, :-1],
                                        self.nodes_linear[2][i](x[:, x.size(1) - k - self.T + self.M - 1:-1, :]))
            recent_data = x[:, -self.M:-1, :]
            h0 = None
            res = None
            for j in range(self.M - 1):
                res, h0 = self.gru(recent_data[:, j:j + 1, :], h0)
            deta = torch.cat([deta, history_deta + attn[:, :, -1:] * res], dim=-1)

        return x_pre + self.W_O(deta[:, :, 1:])
