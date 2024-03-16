# -*- coding:utf-8 -*-
# @project: GSA-forecastor
# @filename: GSAForecaster.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/14 18:56
import torch
import torch.nn as nn

"""
嵌入层在数据变化后必然也会带来mask的变化，新的数据会影响mean和precison
所以mask在每次预测和训练都要重新算
"""


class GSAForecaster(nn.Module):
    """
    标准Encoder-Decoder模型
    """

    def __init__(self, encoder, decoder, graph_embed, aux_embed, pos_embed):
        super(GSAForecaster, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.graph_embed = graph_embed
        self.aux_embed = aux_embed
        self.pos_embed = pos_embed

    def forward(self, x, aux, pre_time):
        """
        seq_to_seq任务
        :param x: shape (batch_size, x_seq_len, d_model) T
        :param aux: shape (batch_size, aux_seq_len, d_aux) T+k
        :param pre_time: 预测时间步
        :return: 返回pre_time时间步的预测结果
        """
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        T = x.size(1)
        pre_seq = torch.zeros(x.size(0), 1, x.size(-1)).to(device)
        for i in range(pre_time):
            aux_emb = self.aux_embed(aux[:, :aux.size(1) - pre_time + i, :])
            pos_emb = self.pos_embed(aux[:, :aux.size(1) - pre_time + i, :])
            x_emb = self.graph_embed(x)
            encode, _ = self.encoder(x_emb[:, :T, :], aux_emb[:, :T, :], pos_emb[:, :T, :])
            encode_combine = encode if i == 0 else torch.cat([encode, x_emb[:, T:, :]], dim=1)
            x_pre, _ = self.decoder(i + 1, encode_combine, encode_combine[:, -1:, :], aux_emb, pos_emb)
            x = torch.cat([x, x_pre], dim=1)
            pre_seq = torch.cat([pre_seq, x_pre], dim=1)
        return pre_seq[:, 1:, :]
