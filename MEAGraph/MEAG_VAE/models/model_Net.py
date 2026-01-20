from abc import ABC

#from graph_ae.Layer import SGAT
from torch_sparse import spspmm, coalesce
#from graph_ae.SAGEConv import SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops
import torch.nn.functional as F
import torch
from torch.nn import Linear
from MEAG_VAE.models.utils import edge_ptr
from MEAG_VAE.models.Encoder_Decoder import Decoder,Encoder

class Net(torch.nn.Module):

    def __init__(self, 
                 input_size, 
                 kernels=3, 
                 pooling_rate=-1, 
                 channels=[128,64,32],
                 edge_reduction_type='score_adaptive'):
        r"""
        Model framework for Graph Autoencoder (GAE).

        Args:
            input_size (int): Number of input features (N_feats).
            kernels (int): Number of kernels (N_kernels) (default: 3).
            pooling_rate (float): Rate for edge reduction (default: -1 for automatic selection).
            channels (list): Encoder dimensions (default: [128, 64, 32]).
            edge_reduction_type (str): Type of edge reduction technique (default: 'score_adaptive').

        Returns:
            tuple: Tuple containing the following elements:
                - z: Decoded features.
                - latent_x: Latent features before decoding.
                - latent_edge: Latent edge indices before decoding.
                - edge_list: List of edge indices stored in the encoder.
        """
        super(Net, self).__init__()
        self.kernels = kernels
        self.depth = len(channels)

        self.pooling_rate=pooling_rate
        self.channels=channels
        self.edge_reduction_type=edge_reduction_type
        self.input_size=input_size
        self.encoder=Encoder(input_size,
                             kernels=self.kernels, 
                             depth=self.depth,
                             pooling_rate=self.pooling_rate,
                             channels=self.channels,
                             edge_reduction_type=self.edge_reduction_type
                             )
        self.decoder=Decoder(input_size, 
                             depth=self.depth, 
                             channels=self.channels)



    def forward(self, x,edge_index):

       
        edge_list = []
        #encoder
        latent_x,latent_edge,edge_list=self.encoder(x,edge_index)
        
    #Uncomment the following lines for adding the Variational layer 
    #    latent_x_mean,latent_x_var,latent_edge,edge_list=self.encoder(f,e)
    #    std = torch.exp(latent_x_var / 2)
    #    eps = torch.randn_like(std)
    #    latent_x_sample = eps.mul(std).add_(latent_x_mean)

        #decoder
        z=latent_x
      #Uncomment for adding the Varational layer
      # z=latent_x_sample
        z=self.decoder(z,edge_list)
        
        #uncomment for returning the Varational layer  
        #return z, latent_x_sample,latent_x_mean, latent_x_var, latent_edge, edge_list
        return z, latent_x, latent_edge, edge_list
    def update_parameters(self,rate):
        """
        Update the hidden size and reset the corresponding layer.

        Args:
            rate (float): New pooling rate.
        """
        self.encoder=Encoder(self.input_size,
                      kernels=self.kernels, 
                      depth=self.depth,
                      pooling_rate=rate,
                      channels=self.channels,
                      edge_reduction_type=self.edge_reduction_type
                      )
    