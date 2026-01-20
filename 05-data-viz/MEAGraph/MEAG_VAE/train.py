import numpy as np
import copy
from tqdm import trange
import torch
from MEAG_VAE.config import cfg
import torch.nn.functional as F
from MEAG_VAE.models.utils import build_graph

def train_cp(model, optimizer,scheduler, device, train_set, valid_set, num_epoch,fixed_rate_l=-1.0,fixed_rate_r=-1.0):
    device=torch.device(cfg.device)
    model=model.to(device)
    losses,val_losses=[],[]
    min_loss=np.inf
    best_model=None
    for e in trange(num_epoch, desc="Training", unit="Epochs"):
        reconstruction_loss = 0
        reconstruction_loss_val = 0
        model.train()
        for data in train_set: 
            data = data.to(device)
            node_mask,x,edge_index=build_graph(data,fixed_rate_l,fixed_rate_r)
            if len(node_mask) == 0: 
                continue   
            z, _,_, _= model(x,edge_index)
            loss = torch.nn.MSELoss()(z, x)     
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reconstruction_loss += loss.item()
        if valid_set:
            model.eval()
            with torch.no_grad():
                for data in valid_set:
                    data = data.to(device)
                    data.x=F.normalize(data.x,p=2,dim=0)
                    z,_,_,_= model(data)
                    mse_loss = torch.nn.MSELoss()(z, data.x)
                    reconstruction_loss_val += mse_loss.item()*data.num_graphs
            reconstruction_loss_val /= len(valid_set.dataset)
            val_losses.append(reconstruction_loss_val)
        reconstruction_loss /= len(train_set)
        losses.append(reconstruction_loss)
        if reconstruction_loss < min_loss:
            min_loss = reconstruction_loss
            best_model = copy.deepcopy(model)        
        scheduler.step()
        print()
        print('Epoch: {:03d}'.format(e))
        print('Training Loss:', reconstruction_loss)
        if valid_set:
            print('Valid Loss:', reconstruction_loss_val)
    return losses,val_losses,min_loss,best_model