from abc import ABC

#from graph_ae.Layer import SGAT
from torch_sparse import spspmm, coalesce
#from graph_ae.SAGEConv import SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.utils import sort_edge_index, add_remaining_self_loops
import torch.nn.functional as F
import torch
from torch.nn import Linear
from MEAG_VAE.models.utils import edge_reduction,edge_reduction_recover,edge_reduction_score_mean,edge_reduction_score_adaptive
from MEAG_VAE.models.MEAG import MEAG
from MEAG_VAE.models.GNN import MEAGlayer,Convlayer
from MEAG_VAE.config import cfg
class Encoder(torch.nn.Module):

    def __init__(self, input_size, kernels,depth, pooling_rate,channels,edge_reduction_type):
        r"""
        Encoder module.

        Args:
            input_size (int): Number of input features.
            kernels (int): Number of kernels.
            depth (int): Number of layers in the encoder.
            pooling_rate (float): Rate for edge reduction.
            channels (list): Feature dimensions of the encoder layers.
            edge_reduction_type (str): Type of edge reduction technique.
        """
        super().__init__()
        self.input_size=input_size
        self.N_kernels = kernels
        self.depth = depth
        self.normalization=True
        self.pooling_rate=pooling_rate
        self.channels=channels
        self.edge_reduction_type=edge_reduction_type
        self.down_list = torch.nn.ModuleList()
        
        #Create Encoder layers
        conv = MEAG(self.N_kernels,self.input_size,channels[0])
        self.down_list.append(conv)
        for i in range(self.depth - 1):
            conv = MEAG(self.N_kernels, channels[i], channels[i + 1])
            self.down_list.append(conv)
        
        # VAE layer
        # -------------------------------------------------------------------------
        # Uncomment the following lines to add the Variational AutoEncoder layer
        # self.mean = Linear(channels[self.depth-1], channels[self.depth-1])
        # self.var = Linear(channels[self.depth-1], channels[self.depth-1])
        # self.reset_parameters()

        # def reset_parameters(self):
        #     """
        #     Reset the parameters of the mean and variance linear layers.
        #     """
        #     self.mean.reset_parameters()
        #     self.var.reset_parameters()
        # --------------------------------------------------------------------------

      
    def forward(self,x,edge_index):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            tuple: Tuple containing latent node features, latent edge indices, and list of edge index at each encoder layer.
        """
        edge_list=[]
        edge_index,_=add_remaining_self_loops(edge_index,num_nodes=x.shape[0])

        for i in range(self.depth):
            edge_list.append(edge_index)
            x,attn=self.down_list[i](x,edge_index)
            x=F.leaky_relu(x)
            x = F.normalize(x, p=2., dim=0)

           # Apply edge reduction techniques
            if self.edge_reduction_type == 'recover':
                edge_index=edge_reduction_recover(edge_index,attn,self.pooling_rate)
            elif self.edge_reduction_type == 'score_mean':
                edge_index=edge_reduction_score_mean(edge_index,attn,self.pooling_rate)
            elif self.edge_reduction_type == 'score_adaptive':
                edge_index=edge_reduction_score_adaptive(edge_index,attn,self.pooling_rate)
            else:
                edge_index=edge_reduction(edge_index,attn,self.pooling_rate)
   
                                           
            if not edge_index.numel():
                edge_index = torch.tensor([[0],[0]], dtype=torch.long,device=cfg.device)
             
          
            edge_index,_=add_remaining_self_loops(edge_index,num_nodes=x.shape[0])

       
        # -------------------------------------------------------------------------
        # Uncomment the following line to add leaky_relu activation to the final output
        # x = F.leaky_relu(x)

        # VAE layer
        # Uncomment the lines below to return the Variational AutoEncoder layer
        # latent_x_mean = self.mean(latent_x)
        # latent_x_var = self.var(latent_x)
        # return latent_x_mean, latent_x_var, latent_edge, edge_list

        # Normalization
        # Uncomment the following lines to apply normalization to the output
        # if self.normalization:
        #     embedding_size = x.size(1)
        #     x = x.reshape(len(ptr_edge) - 1, -1, embedding_size)
        #     x = F.normalize(x, p=2., dim=1)
        #     x = x.reshape(-1, embedding_size)
        # -------------------------------------------------------------------------
        
        latent_x,latent_edge=x,edge_index
        return latent_x,latent_edge,edge_list

class Decoder(torch.nn.Module):

    def __init__(self,
                 input_size,
                 depth,
                 channels):
        """
        Decoder module.

        Args:
            input_size (int): Number of input features.
            depth (int): Number of layers in the decoder.
            channels (list): Feature dimensions of the decoder layers.
        """
        super().__init__()
        self.depth=depth
        self.channels=channels
        self.input_size=input_size

        self.up_list=torch.nn.ModuleList()
        for i in range(self.depth-1):
            conv=Convlayer(channels[self.depth-i-1],channels[self.depth-i-2])
            self.up_list.append(conv)
        self.up_list.append(Convlayer(channels[0],self.input_size))
        
        #A final linear layer, also can be used when VAE switched on 
        self.hidden_to_out=Linear(input_size,input_size)
      
        self.reset_parameters()

    def reset_parameters(self):

        self.hidden_to_out.reset_parameters()
     
    
    def forward(self,z,edge_list):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Latent node features.
            edge_list (list): List of edge indices.

        Returns:
            torch.Tensor: Reconstructed node features.
        """
        for i in range(self.depth):
            index=self.depth-i-1
            z=self.up_list[i](z,edge_list[index])

            if i < self.depth - 1:

                z = F.leaky_relu(z)
        #z = F.normalize(z, p=2., dim=0)
        #z=torch.relu(z)
        
        z=self.hidden_to_out(z)

        return z
