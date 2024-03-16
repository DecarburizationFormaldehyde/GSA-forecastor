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
from utils.loss_utils import *

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#optian the configure from yaml
with open('Experiment_config.yaml', 'r') as f :
    config = yaml.load(f,Loader=yaml.FullLoader)

# optian the datasetsLoader
hour_train_loader, weather_train_loader, hour_test_loader, weather_test_loader, scale=final_get_dataloaders(config['start_year'],config['start_month'],config['end_year'],config['end_month'])


# test
def test(load_state=True,model_loc="",return_graphs=False):
    if load_state:
        model.load_state_dict(torch.load(model_loc))
      
    with torch.no_grad():
        model.eval()        
        predictions = []
        values = []
        
        for batch,batch_a in zip(hour_test_loader,weather_test_loader):
            x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
            x_a_batch, _ = batch_a[0].float().to(device), batch_a[1].float().to(device)

            if return_graphs: 
                y_hat = model(x_batch.float().to(device),x_a_batch.float().to(device),step)
                # 待补充attention权重图形化部分
            else:
                y_hat = model(x_batch.float().to(device),x_a_batch.float().to(device),step)
            
            y_hat = y_hat.cpu().detach().numpy()
            predictions.append(y_hat)
            values.append(y_batch.cpu().detach().numpy())            

    return predictions, values


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


# Training
step=3
n_epochs=3


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

niu=2
criterion =  SpeicalLoss(nn.MSELoss().to(device),mape_loss().to(device),niu)
optimizer=optim.Adam(model.parameters(),lr=0.001)


loggertoE=get_logger('output/epoch.log')
t0 = time.time()
predictions, values = test(model_loc='output/output_model/last_run.pt',load_state = False)
t1 = time.time()
inf_time = t1-t0
get_indicators(values,predictions,loggertoE)

