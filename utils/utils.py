import copy
import torch.nn as nn
import logging
import random

def clones(module,N):
    """
        clone the module N to N layers
        :param module:
        :param N:
        :return: N layer module
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

