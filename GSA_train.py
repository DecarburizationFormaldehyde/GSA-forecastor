import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from utils.utils import *
from utils.data_utils import dataloader

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model.GSA import GSA

with open('Experiment_config.yaml', 'r') as f :
    config = list(yaml.load_all(f))[0]

def train(model):
    model.train()
    batch_losses = []
    