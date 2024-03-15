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


def gen_sparse_mask(input_feature, output_feature, graph_dependency, reserve):
    assert input_feature == len(graph_dependency), \
        "请确保输入神经元数量与图依赖性数量相等"
    assert input_feature <= output_feature, \
        "输入神经元数量应小于等于输出神经元数量"
    assign_index = assign_neurons(input_feature, output_feature)

    weight_mask = np.zeros((input_feature, output_feature))
    for row in range(input_feature):
        assign_indices = [assign_index[i] for i, dep in enumerate(graph_dependency[row] == 1) if dep]
        for indices in assign_indices:
            for index in indices:
                weight_mask[row, index] = 1
    return torch.from_numpy(weight_mask).float().t() if not reserve else \
            torch.from_numpy(weight_mask).float()


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weight=weight.to(device)
        mask=mask.to(device)
        bias=bias.to(device)
        if mask is not None:
            weight = weight * mask
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2,-1).matmul(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=1)

        return grad_input, grad_weight, grad_bias, grad_mask


class SparseLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, graph_dependency=None, reserve=False):
        super(SparseLinear, self).__init__()
        self.input_features = input_features if not reserve else output_features
        self.output_features = output_features if not reserve else input_features

        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            self.register_parameter('bias', None)

        self.init_params()
        if graph_dependency is not None:
            mask = gen_sparse_mask(input_features, output_features, graph_dependency, reserve)
            mask = mask.clone().detach()
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = None

    def init_params(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.uniform_(self.bias, a=0, b=1)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features, self.bias is not None, self.mask is not None
        )


# class SparseLinear(nn.Module):
#     def __init__(self, input_feature, output_feature, graph_dependency=None, reserve=False):
#         super(SparseLinear, self).__init__()
#         if graph_dependency is not None:
#             self.sparse_mask = gen_sparse_mask(input_feature, output_feature, graph_dependency, reserve)
#         else:
#             self.sparse_mask = torch.ones((input_feature, output_feature))
#         # 创建全连接层，并根据稀疏性掩码调整权重
#         self.fc1 = nn.Linear(input_feature, output_feature, bias=False)
#         self.fc1.weight.data *= self.sparse_mask
#
#     def forward(self, x):
#         return self.fc1(x)


