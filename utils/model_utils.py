# -*- coding:utf-8 -*-
# @project: git_project
# @filename: model_utils.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/15 11:29
import torch.nn as nn
import torch
from model.GSA import GSAForecaster
from model.encoder import Encoder, EncoderLayer, GSAFilter
from model.decoder import Decoder, DecoderLayer, GSAPredict
from model.embeddings import GraphEmbeddings, AuxEmbeddings, PosEmbeddings
from model.sparse_linear import SparseLinear
from model.ffd import FFD



def getGSA(
        nodes_size,
        a_dim,
        h,
        d_model,
        d_aux,
        d_pos,
        d_hidden,
        num_encoder_layers,
        num_decoder_layers,
        num_gru_layers,
        M_1,
        M_2,
        M,
        T,
        graph_dependency,
        dropout=0.01
):
    gsa_filter = GSAFilter(h=h, d_model=d_model, d_aux=d_aux, d_pos=d_pos,
                           M_1=M_1, M_2=M_2, T=T, graph_dependency=graph_dependency)
    assert d_model % h == 0, "d_model must be a multiple of h"
    gru = nn.GRU(input_size=d_model, hidden_size=d_model // h, num_layers=num_gru_layers, batch_first=True)
    gsa_pre = GSAPredict(h=h, d_model=d_model, d_aux=d_aux, d_pos=d_pos,
                         M=M, T=T, gru=gru, graph_dependency=graph_dependency)
    graph_dependency_repeat = graph_dependency.repeat(h, h)
    encoder_ffd = FFD(d_model=d_model, d_hidden=d_hidden, graph_dependency=graph_dependency_repeat)
    decoder_ffd = FFD(d_model=d_model, d_hidden=d_hidden, graph_dependency=graph_dependency_repeat)
    encoder_layer = EncoderLayer(gsa_filter=gsa_filter, ffd=encoder_ffd)
    decoder_layer = DecoderLayer(gsa_pre=gsa_pre, ffd=decoder_ffd)
    de_embed = SparseLinear(nodes_size, d_model, graph_dependency=graph_dependency, reserve=True)
    model = GSAForecaster(
        Encoder(encoder_layer=encoder_layer, N=num_encoder_layers),
        Decoder(decoder_layer=decoder_layer, N=num_decoder_layers, de_embed=de_embed),
        GraphEmbeddings(nodes_size=nodes_size, d_model=d_model, graph_dependency=graph_dependency),
        AuxEmbeddings(a_dim=a_dim, d_aux=d_aux),
        PosEmbeddings(d_pos=d_pos, dropout=dropout)
    )
    return model


