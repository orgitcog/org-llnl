
from torch_geometric.nn.pool.topk_pool import topk, filter_adj

from torch_geometric.nn import global_max_pool as g_pooling
from torch_geometric.utils import sort_edge_index
from torch_sparse import spspmm
import torch.nn.functional as f
from torch.nn import Parameter,Linear
import torch

from torch_geometric.nn import SAGEConv
from MEAG_VAE.models.GNN import MEAGlayer


class MEAG(torch.nn.Module):
    r"""
    MEAG framework for multi-kernel GNN layers.

    Args:
        kernels (int): Number of kernels.
        in_channel (int): Size of input features.
        out_channel (int): Size of output features.

    Returns:
        tuple: Tuple containing the output features and averaged attention weights.
    """
    def __init__(self, kernels, 
                 in_channel, 
                 out_channel):


        super().__init__()
        self.N_kernels = kernels
        self.in_channel = in_channel
        self.out_channel = out_channel

        # Create a list of MEAGlayer modules
        self.meag_list = torch.nn.ModuleList()

        for i in range(self.N_kernels):
            self.meag_list.append(MEAGlayer(in_channel, out_channel))
        
        self.lin_feats = Linear(out_channel, out_channel) 
        self.reset_parameters()
              
    def reset_parameters(self):

        for conv in self.meag_list:
            conv.reset_parameters()
        self.lin_feats.reset_parameters()

    def forward(self, x, edge_index):
  
        attention_list = []
        feature_list=None
        # Apply each MEAGlayer to the input features
        for conv in self.meag_list:
            # The kernels (conv) are applied independently to the input features (x).
            # The output features from different kernels are combined later.
            feature, attn = conv(x, edge_index) 
            if feature_list is None:
                feature_list = f.leaky_relu(feature)
            else:
                feature_list += f.leaky_relu(feature)
            attention_list.append(attn)

       
        x_ave=feature_list/self.N_kernels  # Average the features from different kernels
        x_ave=self.lin_feats(x_ave)
     
        attention_list = torch.stack(attention_list, dim=1) # Stack attention weights from different kernels

        if attention_list.shape[1] > 1:
            att_ave = torch.mean(attention_list, dim=1) # Average the attention weights

        return x_ave,att_ave