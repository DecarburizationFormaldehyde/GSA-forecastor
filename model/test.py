# -*- coding:utf-8 -*-
# @project: git_project
# @filename: test.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/15 12:50
import calendar

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from model.sparse_linear import SparseLinear
from utils.data_utils import dateToStr
from utils.model_utils import make_model


def get_weather_data(start_year: int, start_month: int, end_year: int, end_month: int):
    weather_data = pd.read_csv('../datasets/all_padding_ready_weather.csv', sep=',',encoding="utf-8")
    filter_data = weather_data.loc[
        (weather_data['year'] * 12 + weather_data['month'] >= start_year * 12 + start_month)
        & (weather_data['year'] * 12 + weather_data['month'] <= end_year * 12 + end_month)
    ]
    matrix = filter_data.values[:, 4:]
    return matrix


def get_data(start_year: int, start_month: int, end_year: int, end_month: int):  # threshold
    matrix = None
    while start_year < end_year or \
            (start_year == end_year and start_month <= end_month):
        yearStr = str(start_year)
        monthStr = dateToStr(start_month)
        data = pd.read_csv(
            '../datasets/hour_data_matrix_' + yearStr + '/hour_data_matrix' + yearStr + '-' + monthStr + '.csv', sep=',',
            encoding="utf-8")
        matrix = data.values[:, 4:] if matrix is None else \
            np.concatenate((matrix, data.values[:, 4:]), axis=0)
        if (start_month + 1) % 12 == 1:
            start_month = 1
            start_year += 1
        else:
            start_month += 1
    return matrix


def get_train_data(batch, sample_len, train_data,weather_data):
    train_mean = train_data.mean(axis=0, keepdims=True)
    train_std = train_data.std(axis=0, keepdims=True)
    train_data = (train_data - train_mean) / (train_std+1e-6)
    weather_mean = weather_data.mean(axis=0, keepdims=True)
    weather_std = weather_data.std(axis=0, keepdims=True)
    weather_data = (weather_data - weather_mean) / (weather_std+1e-6)
    indices = np.arange(0,train_data.shape[0] - sample_len + 1,sample_len)
    np.random.shuffle(indices)
    nodes_samples = torch.zeros((1, sample_len, train_data.shape[1]))
    aux_samples = torch.zeros((1, sample_len, weather_data.shape[1]))
    count = 0
    index = 0
    for i in indices:
        node_sample = torch.from_numpy(train_data[i:i + sample_len]).float()
        nodes_samples = torch.cat((nodes_samples, node_sample.unsqueeze(0)), dim=0)
        aux_sample = torch.from_numpy(weather_data[i:i + sample_len]).float()
        aux_samples = torch.cat((aux_samples, aux_sample.unsqueeze(0)), dim=0)
        count += 1
        index += 1
        if count == batch:
            print("第{}个batch".format(index // batch))
            yield nodes_samples[1:, :, :], aux_samples[1:, :, :]
            nodes_samples = torch.zeros((1, sample_len, train_data.shape[1]))
            aux_samples = torch.zeros((1, sample_len, weather_data.shape[1]))
            count = 0


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
    loss = nn.MSELoss()
    train_data_samples = get_data(2011, 1, 2011, 1)
    weather_data_samples = get_weather_data(2011, 1, 2011, 1)
    batch_size = 4
    loss_list = []
    for node_data, weather_data in get_train_data(batch_size, 24 * 7 + 3, train_data_samples, weather_data_samples):
        train_data = node_data[:, :T, :]
        test_data = node_data[:, T:, :]
        predict = model(train_data, weather_data, 3)
        l = loss(predict, test_data)
        l.backward()
        loss_list.append(l)
        print(l)
        break
    print(loss_list)
