import copy
import torch.nn as nn


def clones(module,N):
    """
        clone the module N times
        :param module:
        :param N:
        :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])