# -*- coding:utf-8 -*-
# @project: git_project
# @filename: test.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/15 12:50
import numpy as np
import torch
import torch.nn as nn
import yaml

from model.sparse_linear import SparseLinear
from utils.model_utils import make_model


def test_sparse_linear():
    x = np.random.random((2, 3, 6))
    x = torch.tensor(x, dtype=torch.float32)
    y = np.random.random((2, 3, 4))
    y = torch.tensor(y, dtype=torch.float32)
    corr_matrix = np.array([[1, 0, 1, 1],
                            [1, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 1],])
    model = SparseLinear(4, 6, graph_dependency=torch.tensor(corr_matrix, dtype=torch.float32),reserve=True)
    pre=model(x)
    loss = nn.MSELoss()
    loss_value = loss(pre, y)
    loss_value.backward()


def test_model():
    corr_matrix = np.load('./test_data/corr_matrix.npy')  # (67, 67)
    corr_matrix[corr_matrix != 0] = 1
    graph_dependency = torch.tensor(corr_matrix, dtype=torch.float32)
    with open('../Experiment_config.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    nodes_size = result['nodes_size']
    a_dim = result['a_dim']
    h = result['h']
    d_model = result['d_model']
    d_aux = result['d_aux']
    d_pos = result['d_pos']
    d_hidden = result['d_hidden']
    num_encoder_layers = result['num_encoder_layers']
    num_decoder_layers = result['num_decoder_layers']
    num_gru_layers = result['num_gru_layers']
    M_1 = result['M_1']
    M_2 = result['M_2']
    M = result['M']
    T = result['T']
    graph_dependency = graph_dependency
    dropout = result['dropout']
    model = make_model(
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
    )