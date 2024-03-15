# -*- coding:utf-8 -*-
# @project: git_project
# @filename: ffd.py
# @author: xiaowei
# @contact: 2117920996@qq.com
# @time: 2024/3/15 11:49

import torch.nn as nn
from model.sparse_linear import SparseLinear


class FFD(nn.Module):
    def __init__(self, d_model, d_hidden, graph_dependency):
        super(FFD, self).__init__()
        self.fc1 = SparseLinear(d_model, d_hidden, graph_dependency=graph_dependency)
        self.relu = nn.ReLU()
        self.fc2 = SparseLinear(d_model, d_hidden, graph_dependency=graph_dependency, reserve=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
