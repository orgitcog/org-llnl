"""
INR model setup and loading module for PruningAMR.

This module provides functionality for loading and initializing different types of
INRs used in the PruningAMR algorithm. It supports multiple INR architectures 
including CT scans, PINNs, and oscillatory functions.

Key functions:
- load_inr(): Main entry point for loading INR models
- load_gpu_model(): Load and convert GPU models to CPU
- load_cpu_model(): Load pre-converted CPU models
- load_standard_model(): Load standard INR checkpoints
- get_params(): Get INR-specific parameters
- setup_gpu_environment(): Configure GPU environment for model loading
"""

import os
import sys
import pickle
import torch
import torch.distributed as     dist
from   torch.nn.parallel import DistributedDataParallel as DDP
from   typing            import Dict, Any, Optional

from   config     import AMRConfig, get_checkpoint_path
from   ID_pruning import pretrained_INR

def setup_gpu_environment(local_rank: int = 0) -> None:
    """
    Set up the GPU environment for distributed training.
    
    Configures PyTorch distributed training environment for loading GPU models.
    Sets up single-node, single-GPU configuration for model conversion.
    
    Args:
        local_rank: The local rank of the process (default: 0)
    """
    os.environ['RANK']        = '0'
    os.environ['WORLD_SIZE']  = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend = 'nccl')

def get_params(inr_key: str) -> Dict[str, Any]:
    """
    Get INR-specific parameters for model initialization.
    
    Returns model parameters based on INR type:
    - CT models: Network architecture, Fourier features, and scaling parameters
    - NS_PINN: Standardization parameters (mean, std) loaded from checkpoint
    - Other models: Empty dictionary (no special parameters)
    
    Args:
        inr_key: The INR key to determine specific parameter values
        
    Returns:
        Dict containing model-specific parameters
    """
    if "CT" in inr_key:
        params = {
            "INR_key":  inr_key,
            "N_dim":          4,  # Number of dimensions
            "N_feat":       128,  # Number of features in the network
            "N_layers":       5,  # Number of layers in the network
            "batch_norm": False,  # Batch normalization for the network
            "sigma_t":      0.1,  # Sigma for time
            "sigma_s":      0.3,  # Sigma for space
            "scale_lac":    0.1,  # Scale for lac
            "prior_regp":   0.0   # Prior regularization parameter
        }
        
        if inr_key in ['CT2_gpu', 'CT3_gpu']:
            params["sigma_s"] = 0.5
        elif inr_key in ['CT6_gpu']:
            params["sigma_s"] = 0.5
            params["scale_lac"] = 0.05
        elif inr_key in ['CT7_gpu']:
            params["sigma_s"] = 5.0
    elif "NS_PINN" in inr_key:
        # Obtain the mean and standard deviation for transforming NS PINN input data.
        mean_std_path = "./checkpoints/NS_PINN/mean_std.pth"
        checkpoint = torch.load(mean_std_path)
        mean = checkpoint["mean"]
        std  = checkpoint['std']

        params = {"INR_key" : inr_key, "mean" : mean, "std" : std}
        
    return params

def load_gpu_model(config: AMRConfig) -> pretrained_INR:
    """
    Load a GPU model and convert it for CPU usage.
    
    Loads a GPU-trained model, extracts weights and parameters, and creates
    a CPU-compatible pretrained_INR object. Saves the converted model as a
    pickle file for future CPU usage.
    
    Args:
        config: The AMR configuration object
        
    Returns:
        pretrained_INR: The loaded and converted model
        
    Note:
        This function requires GPU access and saves a pickle file for CPU usage.
        After running, use the corresponding '_cpu' INR key for CPU execution.
    """
    import dinr.nnets.model as model
    import dinr.scripts.recon_lite as recon_lite
    
    device = 'cuda'
    savedevice = 'cpu'
    local_rank = 0
    
    # Set up environment
    setup_gpu_environment(local_rank)
    
    # Get model parameters
    params = get_params(config.inr_key)
    
    # Initialize model
    nn_init = model.NNModel(
        params["N_dim"],
        params["N_feat"],
        params["N_layers"],
        params["batch_norm"],
        params["sigma_t"],
        params["sigma_s"],
        params["scale_lac"],
        params["prior_regp"]
    ).to(device)
    
    nn_init_ddp = DDP(nn_init, device_ids=[local_rank])
    
    # Set up inference
    nn_inference = recon_lite.CuminInfer(
        local_rank=local_rank,
        rank=os.environ['RANK'],
        world=None,
        device=device,
        dims=params["N_dim"],
        batch_size=None,
        nn_model=nn_init_ddp,
        rec_fpath=None
    )
    
    # Load checkpoint
    checkpoint_path = get_checkpoint_path(config.inr_key)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    nn_inference.load(checkpoint['nn_model'])
    loaded_gauss = checkpoint['nn_model']['module.gauss']
    NN = nn_inference.nn_model
    NN.eval()
    
    # Update NN lists
    NN.weights_list = []
    NN.bias_list = []
    NN.act_list = []
    
    # Add hidden layers
    for i in range(params["N_layers"] + 1):
        module_weight_key = f'module.fcnn.nn.{2*i}.weight'
        module_bias_key = f'module.fcnn.nn.{2*i}.bias'
        NN.weights_list.append(checkpoint['nn_model'][module_weight_key].to(savedevice))
        NN.bias_list.append(checkpoint['nn_model'][module_bias_key].to(savedevice))
        if i < params["N_layers"]:
            NN.act_list.append(torch.nn.SiLU().to(savedevice))
        else:
            NN.act_list.append(torch.nn.Identity().to(savedevice))
    
    # Create and save pretrained_INR
    INR = pretrained_INR(
        NN.weights_list,
        NN.bias_list,
        NN.act_list,
        transform_input  = True,                        # Transform input to the network
        transform_output = True,                        # Transform output from the network
        params           = params,                      # Parameters for the network
        gauss            = loaded_gauss.to(savedevice)  # Gaussian parameters
    )
    
    pkl_filename = config.inr_key[:-4] + '_INR.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(INR, file)
    
    print(f"Saved pretrained_INR object to {pkl_filename}")
    print("Exiting. Run the script again using INR_key='CT[#]_cpu' on a CPU machine.")
    return INR

def load_cpu_model(config: AMRConfig) -> pretrained_INR:
    """
    Load a pre-converted CPU model from pickle file.
    
    Loads a previously converted CPU model from a pickle file created by
    load_gpu_model(). This is the standard way to load GPU models on CPU.
    
    Args:
        config: The AMR configuration object
        
    Returns:
        pretrained_INR: The loaded CPU model
    """
    pkl_filename = config.inr_key[:-4] + '_INR.pkl'
    with open(pkl_filename, "rb") as file:
        INR = pickle.load(file)
        INR.inr_key = config.inr_key

        return INR

def load_standard_model(config: AMRConfig) -> pretrained_INR:
    """
    Load a standard INR model from checkpoint file.
    
    Loads INR models that are already in the pretrained_INR format,
    including oscillatory functions and other standard INR types.
    Handles input transformations for CT and NS_PINN models.
    
    Args:
        config: The AMR configuration object
        
    Returns:
        pretrained_INR: The loaded model
    """
    checkpoint_path = get_checkpoint_path(config.inr_key)
    model_lists = torch.load(checkpoint_path)

    if ("NS_PINN" in config.inr_key) or ("CT" in config.inr_key):
        params = get_params(config.inr_key)
        transform_input = True
    else:
        params = {}
        transform_input = False

    return pretrained_INR(
        model_lists['weights_list'],
        model_lists['bias_list'],
        model_lists['act_list'],
        transform_input=transform_input,
        transform_output=False,
        w0=1,
        params = params
    )

def load_inr(config: AMRConfig) -> pretrained_INR:
    """
    Load an INR model based on the configuration.
    
    Main entry point for loading INR models. Automatically selects the appropriate
    loading method based on the INR key:
    - '_gpu' keys: Load and convert GPU models
    - '_cpu' keys: Load pre-converted CPU models  
    - Other keys: Load standard INR checkpoints
    
    Args:
        config: The AMR configuration object
        
    Returns:
        pretrained_INR: The loaded model ready for PruningAMR
    """
    if '_gpu' in config.inr_key:
        return load_gpu_model(config)
    elif '_cpu' in config.inr_key:
        return load_cpu_model(config)
    else:
        return load_standard_model(config) 
