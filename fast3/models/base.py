import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torch_geometric.data import Batch
from pdb import set_trace


class BaseModel(nn.Module):

    def __init__(self, in_channels, out_channels=1, loss_fn=None, estlabel=False):
        super(BaseModel, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_fn = loss_fn
        self.estLabel = estlabel
            

    def get_optimizer_params(self, configs):
        configs["params"] = self.parameters()
        return configs

    def initialize(self):
        pass

    def forward(self, inputs):
        if not torch.cuda.is_available():
            inputs = Batch.from_data_list(inputs)

        if self.estLabel:
            outputs, pred_labels = self._forward(inputs)
        else:
            outputs = self._forward(inputs)
        if self.training:
            if self.estLabel:
                losses = self.losses_wLabels(outputs, inputs.y,  inputs.label.view_as(pred_labels), pred_labels)
            else:
                losses = self.losses(outputs, inputs.y)
            
            return losses
        
        if self.estLabel:
            return outputs, pred_labels
        
        return outputs

    def losses(self, predictions, targets):
        loss_dict = {}
        for loss_name, loss_fn in self.loss_fn.items():
            if loss_name.startswith(("mse", "l2", "bce")):
                loss_dict[loss_name] = loss_fn(predictions, targets)
            else:
                raise ValueError("Unknown loss: {}".format(loss_name))

        return loss_dict
        
    def losses_wLabels(self, predictions, targets, labels, predicted_labels):
        loss_dict = {}
        
        for loss_name, loss_fn in self.loss_fn.items():
            if loss_name.startswith(("mse", "l2")):
                loss_dict[loss_name] = loss_fn(predictions, targets)
            elif loss_name.startswith(("bce", "ce")):
                loss_dict[loss_name] = loss_fn(predicted_labels, labels)
            else:
                raise ValueError("Unknown loss: {}".format(loss_name))

        return loss_dict
    



class mBaseModel(nn.Module):

    def __init__(self, in_channels, out_channels=1, loss_fn=None, estpose=False):
        super(mBaseModel, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_fn = loss_fn
        self.estpose = estpose
            

    def get_optimizer_params(self, configs):
        configs["params"] = self.parameters()
        return configs

    def initialize(self):
        pass

    def forward(self, inputs):
        if not torch.cuda.is_available():
            inputs = Batch.from_data_list(inputs)
        
        
        outputs, pos_l, pos_p = self._forward(inputs)
        
        if self.training:
            losses = self.losses(outputs, inputs, pos_l, pos_p)
            return losses
        
        return outputs

    def losses(self, predictions, targets, pos_l=None, pos_p=None):
        loss_dict = {}
        

        for loss_name, loss_fn in self.loss_fn.items():
            try:
                if loss_name == 'pmse':                    
                    loss_dict[loss_name] = loss_fn(targets.pos[targets.edge_index[0,:].unique()], pos_p) + \
                        loss_fn(targets.pos[targets.edge_index[1,:].unique()], pos_l)
                else:
                    loss_dict[loss_name] = loss_fn(predictions, targets.y)
            except:
                raise ValueError("Unknown loss: {}".format(loss_name))

        return loss_dict

