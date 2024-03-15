import copy

import torch
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


def attention(similarity):
    exp_similarity = torch.exp(similarity)
    exp_sum_similarity = torch.sum(exp_similarity, dim=-1,keepdim=True)
    return exp_similarity / (exp_sum_similarity + 1e-6)




# save the console print into the log
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# unit test
logger=get_logger("../output/test.log")
logger.info('start training!')

for epoch in range(50):
    loss = random.randint(0, 100)
    acc =  random.randint(0, 100)
    logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , 50, loss, acc ))