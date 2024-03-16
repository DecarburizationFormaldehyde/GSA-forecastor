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
hour_train_loader, weather_train_loader, hour_test_loader, weather_test_loader, scale=final_get_dataloaders(config['start_year'],config['start_month'],config['end_year'],config['end_month'])

# train
def train(model,logger):
    model.train()
    batch_losses = []
    i=0
    for batch,batch_a in zip(hour_train_loader,weather_train_loader):
        t0 = time.time()  
        x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
        x_a_batch, _ = batch_a[0].float().to(device), batch_a[1].float().to(device)

        y_hat=model(x_batch.float().to(device),x_a_batch.float().to(device),step)
        
        loss = criterion(y_batch,y_hat.float().to(device))
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        t1=time.time()
        batch_losses.append(loss.item())
        # 此处有问题
        logger.info(f"[{i+1}/{x_batch.shape[0]}] Training loss: {batch_losses[-1]:.4f} \t Time: {t1-t0:.2f}")
        i=(i+1)%16
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
    logger1 = get_logger('output/epoch'+str(epoch)+'.log')
    batch_losses = train(model,logger1)
    t1 = time.time()
        
    train_losses.append(np.mean(batch_losses))
    train_time.append(t1-t0)

    logger=get_logger('output/epoch.log')
    logger.info(f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f} \t Time: {t1-t0:.2f}")

torch.save(model.state_dict(), "output/output_model/last_run.pt")