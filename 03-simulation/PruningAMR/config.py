"""
Configuration module for PruningAMR algorithm.

This module provides configuration management for the PruningAMR system, including:
- AMRConfig dataclass for algorithm parameters
- INR model path definitions and lookup functions
- Mesh file selection based on INR type
- Output directory management

Key components:
- AMRConfig: Main configuration class with all algorithm parameters
- INR_PATHS: Dictionary mapping INR keys to checkpoint file paths
- get_checkpoint_path(): Retrieves checkpoint path for given INR key
- get_mesh_file(): Selects appropriate mesh file for INR type
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os
from pathlib import Path


@dataclass
class AMRConfig:
    """
    Configuration class for PruningAMR algorithm parameters.
    
    This dataclass contains all configuration parameters for the PruningAMR algorithm,
    including INR model selection, refinement control, error computation, and output options.
    
    Parameter Effects:
    - threshold: Controls refinement sensitivity (smaller = more refinement)
    - prop_threshold: Controls pruning-based refinement (smaller = more refinement)  
    - epsilon: Controls ID pruning aggressiveness (smaller = less pruning = more refinement)
    
    Attributes:
        inr_key:                INR model identifier
        device:                 Computing device ('cpu' or 'cuda')
        time_slice:             Time slice for 4D INRs
        max_it:                 Maximum refinement iterations
        max_dofs:               Maximum degrees of freedom
        num_samples:            Samples for ID pruning
        error_check_samples:    Samples for error computation
        error_threshold:        Error threshold for refinement
        prop_threshold:         Proportion (of pruned neurons) threshold for refining
        epsilon_for_pruning:    Epsilon for ID pruning
        original:               Use uniform refinement
        avg_error:              Use BasicAMR instead of PruningAMR
        paraview:               Generate ParaView output
        num_uniform_ref:        Uniform refinement iterations before adaptive
    """

    # INR settings
    inr_key: str
    device: str = "cpu"
    time_slice: float = 1.0

    # AMR parameters
    max_it: int = 4  # Maximum number of iterations
    max_dofs: int = 5000  # Maximum number of degrees of freedom
    num_samples: int = 256  # Number of samples for ID
    error_check_samples: int = 128  # Number of samples for error check
    error_threshold: float = 1e-3  # Error threshold for refinement
    prop_threshold: float = 0.2  # proportion threshold for pruning
    epsilon_for_pruning: float = 1e-3  # Epsilon for pruning

    # Flags
    original: bool = False
    avg_error: bool = False
    paraview: bool = False
    num_uniform_ref: int = 2

    def __post_init__(self):
        """
        Set up derived attributes after initialization.
        
        Initializes computed attributes including output directory paths,
        time slice string formatting, and error key for directory naming.
        """
        self.output_dir = os.path.join(os.getcwd(), "output")
        self.time_slice_str = (
            f"_tslice_{self.time_slice}"
            if "CT" in self.inr_key #or "PINN" in self.inr_key
            else ""
        )
        self.error_key = "avg_error/" if self.avg_error else ""

    @property
    def experiment_dir(self) -> str:
        """
        Get the experiment directory path.
        
        Returns a unique directory path based on configuration parameters,
        including INR key, error thresholds, sampling parameters, and time slice.
        Different configurations will have different directory names for organization.
        """
        if self.original:
            return os.path.join(
                self.output_dir, f"{self.inr_key}_original{self.time_slice_str}"
            )

        return os.path.join(
            self.output_dir,
            self.inr_key,
            self.error_key,
            f"thrs_{self.error_threshold}"
            f"_prop_thrs_{self.prop_threshold}"
            f"_eps_{self.epsilon_for_pruning}"
            f"_nsamp_{self.num_samples}"
            f"_errsamp_{self.error_check_samples}"
            f"{self.time_slice_str}",
        )

    def create_output_dirs(self):
        """
        Create necessary output directories.
        
        Creates the main output directory and experiment-specific directory
        for storing results, meshes, and analysis files.
        """
        os.makedirs(os.path.join(self.output_dir, self.inr_key), exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)


# Dictionary mapping INR keys to checkpoint file paths
INR_PATHS: Dict[str, str] = {
    "hol": "./checkpoints/high_oscillation_long_",
    "wfm": "./checkpoints/wave_front_mild_",
    "wfs": "./checkpoints/wave_front_steep_",
    "hof": "./checkpoints/high_oscillation_full_",
    "hos": "./checkpoints/high_oscillation_small_",

    "CT2_gpu": "./CT/CT/Asynchronous4DCT_Deben_v202305_2/launch_sigt0_1_sigs0_5/ckpt_latest.pth",
    "CT2_cpu": "./CT/CT/Asynchronous4DCT_Deben_v202305_2/launch_sigt0_1_sigs0_5/ckpt_latest.pth",

    "CT5_gpu": "./CT/CT/CT_4D_MPM_512_output/S05_700/out_cumin_parallel/launch_sigt0_1_sigs0_3/ckpt_latest.pth",
    "CT5_cpu": "./CT/CT/CT_4D_MPM_512_output/S05_700/out_cumin_parallel/launch_sigt0_1_sigs0_3/ckpt_latest.pth",
    
    'NS_PINN'  : './checkpoints/NS_PINN/NS_PINN_lists.ckpt'
}


def get_checkpoint_path(inr_key: str) -> str:
    """
    Get the checkpoint path for a given INR key.
    
    Looks up the INR key in the INR_PATHS dictionary and returns the full
    checkpoint file path. For PINN models, returns the path directly.
    For other models, appends 'lists.ckpt' to the base path.

    Args:
        inr_key: The key to look up in the INR dictionary

    Returns:
        str: The complete checkpoint file path

    Raises:
        KeyError: If the INR key is not found in the dictionary
    """
    if inr_key not in INR_PATHS:
        raise KeyError(f"Unknown INR key: {inr_key}")

    path = INR_PATHS[inr_key]
    if "PINN" in inr_key:
        return path
    return path + "lists.ckpt"


def get_mesh_file(inr_key: str) -> str:
    """
    Get the appropriate mesh file for a given INR key.
    
    Selects the appropriate mesh file based on the INR model type:
    - NS_PINN: Uses meshes/NS_PINN_2x2x10_z0_20.mesh
    - PINN models: Uses meshes/PINN.mesh  
    - CT models: Uses meshes/box-hex.mesh
    - Other models: Uses meshes/rectangle.mesh

    Args:
        inr_key: The INR key to determine which mesh file to use

    Returns:
        str: Path to the appropriate mesh file
    """
    if inr_key == "NS_PINN":
        return os.path.join(os.getcwd(), "meshes", "NS_PINN_2x2x10_z0_20.mesh")
    elif "PINN" in inr_key:
        return os.path.join(os.getcwd(), "meshes", "PINN.mesh")
    elif any(
        key in inr_key for key in ["CT1", "CT2", "CT3", "CT4", "CT5", "CT6", "CT7"]
    ):
        return os.path.join(os.getcwd(), "meshes", "box-hex.mesh")
    else:
        return os.path.join(os.getcwd(), "meshes", "rectangle.mesh")
