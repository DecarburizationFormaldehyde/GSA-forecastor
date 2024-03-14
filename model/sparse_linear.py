# -*- coding:utf-8 -*-
# @project: git_project
# @filename: sparse_linear.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/14 20:37
import copy

import numpy as np
import torch
import torch.nn as nn


def assign_neurons(input_feature, output_feature, method='uniform'):
    """
    输出神经元分配给输入神经元
    :param input_feature: 输入神经元个数
    :param output_feature: 输出神经元个数
    :param method: 分配模式
    :return: 分配序号列表
    """
    mean_num = output_feature // input_feature
    remainder = output_feature % input_feature
    indices = np.arange(output_feature)
    assign_index = []
    if method == 'uniform':  # 平均分配 {[1,2],[3],[4]} 输出: 4个, 输入: 3个
        index = 0
        for i in range(remainder):
            assign_index.append(indices[index:index + mean_num + 1])
            index += mean_num + 1
        for i in indices[index::mean_num]:
            assign_index.append(indices[i:i + mean_num])
    if method == 'tail':  # 留尾分配 {[1],[2],[3,4]} 输出: 4个, 输入: 3个
        assign_index = [indices[i:i + mean_num] for i in indices[::mean_num]]
        if len(assign_index[-1]) != mean_num:
            last_index = np.hstack([assign_index[-2], assign_index[-1]])
            assign_index = copy.deepcopy(assign_index[:-2])
            assign_index.append(last_index)
    return assign_index


def gen_sparse_mask(input_feature, output_feature, graph_dependency):
    assign_index = assign_neurons(input_feature, output_feature)
    assert input_feature == len(graph_dependency), \
        "请确保输入神经元数量与图依赖性数量相等"
    weight_mask = np.zeros((input_feature, output_feature))
    for row in range(input_feature):
        assign_indices = [assign_index[i] for i, dep in enumerate(graph_dependency[row] == 1) if dep]
        for indices in assign_indices:
            for index in indices:
                weight_mask[row, index] = 1
    return torch.from_numpy(weight_mask).float()


class SparseLinear(nn.Module):
    def __init__(self, input_feature, output_feature, graph_dependency=None, reserve=False):
        super(SparseLinear, self).__init__()
        if graph_dependency is not None:
            self.sparse_mask = gen_sparse_mask(output_feature, input_feature, graph_dependency)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
