import torch.nn as nn

from utils.utils import clones


class Decoder(nn.Module):
    def __init__(self, decoder_layer, N, de_embed):
        super(Decoder, self).__init__()
        self.layers = clones(decoder_layer, N)
        self.de_embed = de_embed

    def forward(self, k, C, D, E, pos):
        for layer in self.layers:
            D = layer(k, C, D, E, pos)
        return self.de_embed(D)


class DecoderLayer(nn.Module):
    def __init__(self, gsa_filter, ffd):
        super(DecoderLayer, self).__init__()
        self.gsa_filter = gsa_filter
        self.ffd = ffd

    def forward(self, k, C, D, E, pos):
        x = self.gsa_filter(k, C, D, E, pos)
        return x + self.ffd(x)


class GSAFilter(nn.Module):
    def __init__(self, h, d_model, d_aux, d_pos, M, T, graph_dependency=None):
        super(GSAFilter, self).__init__()
        assert d_model % h == 0 and d_aux % h == 0 and d_pos % h == 0, \
            "d_model, d_aux, d_pos must be multiple of h"

        self.d_k = d_model // h
        self.d_a = d_aux // h
        self.d_p = d_pos // h

        self.h = h
        self.M = M
        self.T = T

        # 线性变换层
        self.nodes_linear = clones(nn.Linear(d_model, self.d_k), h * 3)
        self.aux_linear = clones(nn.Linear(d_aux, self.d_a), h * 2)
        self.pos_linear = clones(nn.Linear(d_pos, self.d_p), h * 2)

        # Wo层
        graph_dependency_repeat = graph_dependency.repeat(h, h)
        self.W_O = nn.Linear(d_model, d_model, )

    def forward(self, k, C, D, E, pos):
        pass
