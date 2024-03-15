import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from utils.utils import *
from utils.data_utils import *
from model.GSA import GSAForecaster
from utils.model_utils import getGSA
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#optian the configure from yaml
with open('Experiment_config.yaml', 'r') as f :
    config = yaml.load(f,Loader=yaml.FullLoader)

# optian the datasetsLoader
train_loader, test_loader, test_loader_one, scaler=get_dataloaders(get_data(config['start_year'],config['start_month'],config['end_year'],config['end_month']))


# train
def train(model):
    model.train()
    batch_losses = []
    for i,batch in enumerate(train_loader):   
        x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
        y_hat=model(x_batch[:,:,:67].float().to(device),x_batch[:,:,67:].float().to(device),step)

        loss = criterion(y_batch,y_hat.float().to(device))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.item())
    return batch_losses



    

# test
def test(test_loader,loadstate=True,medel_loc="",return_graphs=False):
    pass


# Model args
nodes_size = config['nodes_size']
a_dim = config['a_dim']
h = config['h']
d_model = config['d_model']
d_aux = config['d_aux']
d_pos = config['d_pos']
d_hidden = config['d_hidden']
num_encoder_layers = config['num_encoder_layers']
num_decoder_layers = config['num_decoder_layers']
num_gru_layers = config['num_gru_layers']
M_1 = config['M_1']
M_2 = config['M_2']
M = config['M']
T = config['T']

corr_matrix = np.load('model/test_data/corr_matrix.npy')  # (67, 67)
corr_matrix[corr_matrix != 0] = 1
graph_dependency = torch.tensor(corr_matrix, dtype=torch.float32).to(device)
# Data args

# train args
train_loss=[]


# Training
step=3
n_epochs=50


model=getGSA(
        nodes_size,
        a_dim,
        h,
        d_model,
        d_aux,
        d_pos,
        d_hidden,
        num_encoder_layers,
        num_decoder_layers,
        num_gru_layers,
        M_1,
        M_2,
        M,
        T,
        graph_dependency,
        dropout=0.01
).to(device)

criterion =  nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

train_losses=[]
train_time=[]

for epoch in range(n_epochs):
    t0 = time.time()
    batch_losses = train(model)
    t1 = time.time()
        
    train_losses.append(np.mean(batch_losses))
    train_time.append(t1-t0)

    logger=get_logger('output/epoch')
    logger.info(f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f} \t Time: {t1-t0:.2f}")

torch.save(model.state_dict(), "output/output_model/last_run.pt")