"""
This module implements the pretrained_INR class, which provides functionality for loading
pre-trained Implicit Neural Representations (INRs) and applying Interpolative Decomposition (ID)
pruning to the INR. This can be used toguide adaptive mesh refinement decisions in the 
PruningAMR algorithm.

Key Features:
- Support for multiple INR architectures (CT scans, PINNs, oscillatory functions)
- Fourier feature transformation option for CT INRs
- Interpolative Decomposition pruning with configurable error tolerance
- Forward pass option for both original and pruned networks
- Model saving and loading capabilities

The ID pruning algorithm works by:
1. Sampling points from the element domain
2. Computing the outputs from each layer of the original network
3. Applying ID to find a low-rank approximation of each layer
4. Constructing pruned weights and biases that maintain approximation accuracy from the low-rank approximations
5. Creating a new pruned network with fewer parameters

This module is essential for the PruningAMR algorithm, providing the neural network
compression techniques that guide mesh refinement decisions.
"""

import torch
import torch.nn as nn
import numpy as np 
import os 
import sys

path = os.getcwd()
par_dir = os.pardir
ID_path = os.path.abspath(os.path.join(path, 'ID_Pruning'))
sys.path.append(ID_path)
ID_path = os.path.abspath(os.path.join(par_dir, 'ID_Pruning'))
sys.path.append(ID_path)
from ID import ID
from error import sample_domain 


class pretrained_INR(nn.Module):
    """
    Pre-trained INR class with ID pruning capabilities.
    
    This class provides a PyTorch module for loading and working with pre-trained INRs,
    including support for different input/output transformations and ID-based pruning.
    The class can be used to create new models from known INR weights or to create
    pruned versions of existing INRs for adaptive mesh refinement.
    
    Args:
        weights_list: List of weight tensors for each layer
        bias_list: List of bias tensors for each layer  
        act_list: List of activation functions for each layer
        transform_input: Whether to apply input transformations (default: True)
        transform_output: Whether to apply output transformations (default: True)
        w0: Frequency parameter for sine activations, if using (default: 30)
        **kwargs: Additional parameters including 'params' for INR-specific settings
        
    Attributes:
        num_layers: Number of layers in the network
        widths: List of layer widths
        Layers: PyTorch ModuleList of linear layers
        Pruned_Layers: PyTorch ModuleList of pruned layers (after ID_prune)
        ks: List of ranks for each pruned layer
        params: Dictionary of INR-specific parameters
        gauss: Gaussian parameters for Fourier features (CT INRs)
        mean, std: Standardization parameters (used forNS-PINN)
    """
    def __init__(self, weights_list, bias_list, act_list, transform_input = True, transform_output = True, w0 = 30, **kwargs):
        super().__init__()
        self.weights_list     = weights_list
        self.bias_list        = bias_list
        self.act_list         = act_list
        self.transform_input  = transform_input
        self.transform_output = transform_output
        try:
            self.params       = kwargs["params"]
        except:
            print("no params passed to pretrained_INR initialization")

        if len(self.weights_list) > 0:
            # Define layers to be linear with given weight, bias, and activation function from inputs
            self.num_layers = len(weights_list)
            self.Layers = []
            self.widths = []
            for i in range(self.num_layers):
                input_size  = self.weights_list[i].shape[1]
                output_size = self.weights_list[i].shape[0]
                self.widths.append(output_size)

                self.Layers.append(torch.nn.Linear(input_size, output_size))
                self.Layers[i].weight = nn.parameter.Parameter(self.weights_list[i])
                self.Layers[i].bias   = nn.parameter.Parameter(self.bias_list[i])
                # ensure that computation graphs aren't built
                self.Layers[i].weight.requires_grad_(False)
                self.Layers[i].bias.requires_grad_(False)
            self.Layers = nn.ModuleList(self.Layers)

        # for use if any layers use sine activation function
        self.w0 = w0

        if transform_input == True:
            print("saving self.inr_key")
            self.inr_key = self.params["INR_key"]
            try:
                if "CT" in self.inr_key:
                    N_dim   = self.params['N_dim']
                    sigma_t = self.params["sigma_s"]
                    sigma_s = self.params["sigma_t"]
                    N_feat  = self.params["N_feat"]
                elif "NS_PINN" in self.params["INR_key"]:
                    self.mean = self.params["mean"]
                    self.std  = self.params["std"]
            except:
                print("params not defined for transforming input")
                exit()
            if "CT" in self.inr_key:
                try:
                    self.gauss = kwargs["gauss"]
                    print("Loaded gauss in")
                except:
                    print("Gauss undefined for fourier features. Creating new ones from random.")

                    assert N_dim >= 2
                    sigma = torch.tensor([sigma_t] + [sigma_s] * (N_dim - 1), dtype = torch.float32)
                    sigma = torch.unsqueeze(sigma, dim = 1)

                    self.gauss = torch.randn((N_dim, N_feat), dtype = torch.float32) * sigma
                    self.gauss = nn.parameter.Parameter(self.gauss, requires_grad = False)

    def forward(self, inputs, n = None, pruned = False):
        """
        Forward pass through the neural network.
        
        This method performs a forward pass through either the original or pruned
        network, with optional input/output transformations and partial layer evaluation.
        
        Args:
            inputs: Input tensor of shape (batch_size, input_dim)
            n: Number of layers to evaluate (zero-indexed). If None, evaluates all layers.
               Must be between 0 and num_layers-1.
            pruned: Whether to use pruned layers (requires ID_prune to be called first)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
            
        Raises:
            SystemExit: If n is out of valid range or pruned=True but no pruned layers exist
            
        Note:
            - Input transformations are applied if transform_input=True
            - Output transformations are applied if transform_output=True and n=num_layers-1
            - The method clones inputs to avoid modifying the original tensor
        """
        X = torch.clone(inputs)
        if self.transform_input:
            X = self.transform_data(X) # transform input

        # n is the number of layers we will evaluate (zero-indexed). If not given, we will evaluate all layers of the network.
        if n == None:
            n = self.num_layers - 1
        elif n > self.num_layers - 1 or n < 0:
            print("Error: n needs to be between 0 and the number of layers - 1 ({})".format(self.num_layers - 1))
            exit()

        size = X.size()
        X = torch.reshape(X, (-1, size[-1]))

        # apply layers to X 
        for i in range(n + 1):
            if pruned == False:
                X = self.Layers[i](X)
            else:
                try:
                    X = self.Pruned_Layers[i](X)
                except:
                    print("Error: Cannot use pruned forward method if pruned layers do not exist.")
                    exit()

            # apply activation function
            X = self.act_list[i](X)

            # transform final output
            if i == (self.num_layers - 1) and self.transform_output == True:
                X = self.output_transform(X)
                
#           X = torch.reshape(X, (*size[:-1], 1)) # this command throws an error
            X.reshape([-1,1])

        return X

    def transform_data(self, inputs):
        """
        Transform input data before passing through the network.
        
        This method applies INR-specific input transformations based on the model type:
        - CT INRs: Creates Fourier features using random Gaussian projections
        - NS-PINN: Standardizes inputs using mean and standard deviation
        - Other INRs: Returns inputs unchanged
        
        Args:
            inputs: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Transformed input tensor
            
        Note:
            - For CT INRs, Fourier features are created using the stored Gaussian parameters
            - For NS-PINN, standardization uses the stored mean and std parameters
            - The transformation type is determined by the INR key in self.params
        """
        if "CT" in self.inr_key:
            ff = torch.matmul(inputs, self.gauss) * 2 * np.pi  # batch, t*x*y*z, Nout
            if len(ff.shape) == 1:
                ff = torch.unsqueeze(ff, dim = 0)

    #        ff1 = torch.cat((torch.cos(ff), torch.sin(ff)), dim = 1)  # batch, t*x*y*z, 2*Nout

            if len(ff.shape) > 2:
                ff = torch.cat((torch.cos(ff), torch.sin(ff)), dim=2)  # batch, t*x*y*z, 2*Nout
            else:
                ff = torch.cat((torch.cos(ff), torch.sin(ff)), dim=1)  # batch, t*x*y*z, 2*Nout

            return ff
        elif "NS_PINN" in self.inr_key:
            inputs = (inputs - self.mean)/self.std
            return inputs

    def output_transform(self, output):
        return output * self.params["scale_lac"]

    def ID_prune(self, input_range, epsilon = 10**-4, num_samples = 256):
        """
        Apply Interpolative Decomposition pruning to the neural network.
        
        This method implements the core ID pruning algorithm used in PruningAMR.
        It samples points from the input domain, computes layer outputs, and applies
        Interpolative Decomposition to find low-rank approximations of each layer.
        The pruned network maintains approximation accuracy while reducing parameters.
        
        The algorithm works by:
        1. Sampling num_samples points from the input_range domain
        2. Computing outputs for each layer on the sampled points
        3. Applying ID to find low-rank approximation of each layer
        4. Constructing pruned weights and biases that maintain accuracy
        5. Creating new pruned layers with reduced parameters
        
        Args:
            input_range: Domain bounds for sampling points (list of [min, max] pairs)
            epsilon: Error tolerance for ID approximation (default: 1e-4)
            num_samples: Number of points to sample from domain (default: 256)
            
        Note:
            - This method modifies the network by creating self.Pruned_Layers
            - The pruned network can be used with forward(pruned=True)
            - Layer ranks are stored in self.ks for analysis
            - Pruning is applied layer by layer, maintaining network structure
            - In order to permanently prune the network, use set_weights_to_pruned()
        """
        # initialize interpolation matrix T_prev to the identity
        input_size = self.weights_list[0].shape[1]
        T_prev = torch.tensor(np.identity(input_size, dtype = np.float32))

        # define inputs
        inputs = sample_domain(input_range, num_samples)
        # initialize pruned weights and biases lists
        try:
            self.pruned_weights
        except:
            empty_list = [None for i in range(self.num_layers)]
            self.pruned_weights = empty_list.copy()
            self.pruned_biases  = empty_list.copy()
            self.Pruned_Layers  = empty_list.copy()

        self.ks = []
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                # compute output from ith layer
                with torch.inference_mode():
                    output_i = self.forward(inputs, n = i, pruned = False)

                # compute ID of output of ith layer, with error tolerance epsilon
                #print(output_i)
                P_I, T, k = ID(output_i, epsilon = epsilon)
                self.ks.append(k)

                # form new weights and biases
                self.pruned_weights[i] = self.weights_list[i][P_I, :] @ T_prev.transpose(0, 1)
                self.pruned_biases [i] = self.bias_list[i][P_I] 

                # set last T to T_prev
                T_prev = T


            # for the last (linear) layer, just multiply T_prev with the old weight/bias matrix
            elif i == self.num_layers - 1:
                self.pruned_weights[i] = self.weights_list[i] @ T_prev.transpose(0, 1) 
                self.pruned_biases[i]  = self.bias_list[i] # this one is already set to the correct tensor

            # add layer to pruned layer list
            output_size, input_size      = self.pruned_weights[i].shape
            self.Pruned_Layers[i]        = torch.nn.Linear(input_size, output_size)
            self.Pruned_Layers[i].weight = nn.parameter.Parameter(self.pruned_weights[i])
            self.Pruned_Layers[i].bias   = nn.parameter.Parameter(self.pruned_biases [i])

        # convert to module list
        self.Pruned_Layers = nn.ModuleList(self.Pruned_Layers)

    def set_weights_to_pruned(self):
        """
        Replace original weights with pruned weights.
        
        This method permanently replaces the original network weights and biases
        with the pruned versions created by ID_prune. This is useful for creating
        a permanently pruned network or for further pruning operations.
        
        Note:
            - This operation is irreversible without reloading the original model
            - The network will use the pruned architecture after this call
            - All subsequent forward passes will use these pruned weights
            - note that it is possible to try to prune the network further after 
              this call by calling ID_prune again
        """
        self.weights_list = nn.ParameterList(self.pruned_weights)
        self.bias_list    = nn.ParameterList(self.pruned_biases)
        self.Layers       = self.Pruned_Layers

    def save(self, save_path):
        """
        Save the INR model to a checkpoint file.
        
        This method saves the complete INR model including weights, biases,
        activation functions, and transformation parameters to a PyTorch checkpoint.
        
        Args:
            save_path: Path where to save the model checkpoint
            
        Note:
            - Saves weights_list, bias_list, and act_list for all layers
            - Includes gauss parameters for CT INRs with Fourier features
            - The saved model can be loaded with the load() method
        """
        save_dict = {}
        save_dict['weights_list']   = self.weights_list
        save_dict['bias_list']      = self.bias_list
        save_dict['act_list']       = self.act_list
        if self.transform_input:
            save_dict['gauss']          = self.gauss

        torch.save(save_dict, save_path)

    def load(self, ckpt_fn, device = 'gpu'):
        """
        Load an INR model from a checkpoint file.
        
        This method loads a complete INR model from a PyTorch checkpoint file,
        including weights, biases, activation functions, and transformation parameters.
        The loaded model is ready for forward passes and pruning operations.
        
        Args:
            ckpt_fn: Path to the checkpoint file to load
            device: Device to load the model on ('gpu' or 'cpu')
            
        Raises:
            SystemExit: If the checkpoint file cannot be loaded
            
        Note:
            - Sets requires_grad=False for all parameters to disable training
        """
        # load checkpt file
        try:
            ckpt = torch.load(ckpt_fn, map_location = device)
            print("Loaded the file to " + device)
        except:
            print("\nDID NOT LOAD CHECKPOINT FILE\n")
            exit()

        # load Gauss for fourier features
        self.gauss = ckpt['gauss']

        # load weights, biases, and activations
        self.weights_list = ckpt['weights_list']
        self.bias_list    = ckpt['bias_list']
        self.act_list     = ckpt['act_list']

        # Define layers to be linear with given weight, bias, and activation function from inputs
        self.num_layers = len(self.weights_list)
        self.Layers = []
        for i in range(self.num_layers):
            input_size  = self.weights_list[i].shape[1]
            output_size = self.weights_list[i].shape[0]

            self.Layers.append(torch.nn.Linear(input_size, output_size))
            self.Layers[i].weight = nn.parameter.Parameter(self.weights_list[i])
            self.Layers[i].bias   = nn.parameter.Parameter(self.bias_list[i])

        self.Layers = nn.ModuleList(self.Layers)

        print("loaded in pretrained model")
