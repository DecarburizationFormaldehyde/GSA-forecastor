import copy
import torch.nn as nn
import logging
import random
import torch

def clones(module,N):
    """
        clone the module N to N layers
        :param module:
        :param N:
        :return: N layer module
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

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


\