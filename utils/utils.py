import copy
import torch.nn as nn
import logging
import random
import torch
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clones(module,N):
    """
        clone the module N to N layers
        :param module:
        :param N:
        :return: N layer module
    """
    return nn.ModuleList([copy.deepcopy(module).to(device) for i in range(N)])

def attention(similarity):
    exp_similarity = torch.exp(similarity)
    exp_sum_similarity = torch.sum(exp_similarity, dim=-1, keepdim=True)
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

def get_indicators(value,prediction,logger):
    mae=mean_absolute_error(value.tolist(), prediction.tolist()),             
    mse= mean_squared_error(value.tolist(), prediction.tolist()),
    r2= r2_score(value.tolist(), prediction.tolist(), multioutput = "variance_weighted")
    logger.info('mse'+f'{np.mean(mse):.3f} ± {np.std(mse, ddof=1):.3f}')
    logger.info('r2'+f'{np.mean(r2):.3f} ± {np.std(r2, ddof=1):.3f}')
