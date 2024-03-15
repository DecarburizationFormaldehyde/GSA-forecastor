import copy

import torch
import torch.nn as nn


def clones(module,N):
    """
        clone the module N times
        :param module:
        :param N:
        :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(similarity):
    exp_similarity = torch.exp(similarity)
    exp_sum_similarity = torch.sum(exp_similarity, dim=-1,keepdim=True)
    return exp_similarity / (exp_sum_similarity + 1e-6)


