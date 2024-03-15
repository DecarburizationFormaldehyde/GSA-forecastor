import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from utils.utils import *
from utils.data_utils import dataloader
from model.GSA import GSAForecaster
from model.decoder import Decoder,GSAFilter
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model.GSA import GSA
#optian the configure from yaml
with open('Experiment_config.yaml', 'r') as f :
    config = list(yaml.load_all(f))[0]

# train
def train(model):
    model.train()
    batch_losses = []
    for i,() in enumerate(trian_loader):
        pass
    


# test
def test(test_loader,loadstate=True,medel_loc="",return_graphs=False):
    pass


# Model args

# Data args

# train args
train_loss=[]

# Save args
savePath='/output'

# Training
step=1
n_epochs=50


    
)
