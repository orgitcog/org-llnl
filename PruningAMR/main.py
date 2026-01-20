"""
PruningAMR: Efficient Visualization of Implicit Neural Representations via Weight Matrix Analysis

This script implements the PruningAMR algorithm for variable resolution visualization of 
Implicit Neural Representations (INRs). The algorithm uses neural network pruning techniques 
based on Interpolative Decomposition (ID) to guide mesh refinement decisions, creating 
efficient representations of various neural network models.

Key Features:
- Supports multiple INR architectures (CT scans, PINNs, oscillatory functions)
- Parallel error computation using multi-threading
- Both uniform and adaptive refinement strategies
- Integration with MFEM for mesh operations
- ParaView visualization support

Algorithm Overview:
1. Load pre-trained INR model and initialize mesh
2. Optionally run uniform refinement iterations
3. For adaptive refinement:
   - if using PruningAMR, apply ID pruning to INR on each element domain
   - if using PruningAMR, compute error between original and pruned networks
   - if using BasicAMR, compute error between mesh approximation and original INR
   - Refine elements exceeding error/proportion thresholds
   - Update mesh and repeat until convergence
4. Save refined meshes, grid functions, and analysis results
5. Save parameters and results to JSON file for analysis and reproducibility

Usage:
    python main.py --INR_key <model_key> [options]
    
See README.md for detailed usage instructions and parameter descriptions.
"""
import os
import torch

# Limit BLAS libraries that honour their own vars
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # OpenBLAS, BLIS
os.environ["MKL_NUM_THREADS"]      = "1"   # Intel MKL, oneAPI
os.environ["NUMEXPR_NUM_THREADS"]  = "1"   # numexpr, pandas eval
os.environ["OMP_NUM_THREADS"]      = "1"

# Optional: tell PyTorch's *own* intra-op pools as well
torch.set_num_threads(1)           # CPU kernels
torch.set_num_interop_threads(1)   # cross-op task dispatcher

import faulthandler, signal, sys
faulthandler.enable()                     # turns on the built-in SIGSEGV printer
# optional: dump the traceback of every thread on Ctrl-\  (SIGQUIT)
faulthandler.register(signal.SIGQUIT, file=sys.stderr)

import numpy as np
import argparse
import json
from os import cpu_count
import mfem.ser as mfem

from config       import AMRConfig, get_checkpoint_path, get_mesh_file
from inr_setup    import load_inr
from mesh_handler import MeshHandler
from refinement   import RefinementWorker, compute_original_element_size
import error
#from ID_pruning   import pretrained_INR

## set random number generator seed
np.random.seed(seed = 3)


def parse_args() -> AMRConfig:
    """
    Parse command line arguments and create AMR configuration.
    
    This function handles all command line arguments for the PruningAMR algorithm,
    including INR model selection, refinement parameters, error thresholds, and
    output options. Arguments are validated and converted into an AMRConfig object.
    
    Key argument categories:
    - INR model selection (--INR_key)
    - type of refinement (--original, --avg_error)
    - Refinement control (--max_it, --max_dofs, --num_uniform_ref)
    - Error computation (--threshold, --prop_threshold, --epsilon)
    - Number of samples (--ID_samples, --error_samples)
    - Output options (--paraview)
    
    Returns:
        AMRConfig: Configuration object containing all parsed parameters
        
    Raises:
        SystemExit: If invalid arguments are provided
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--original',       action='store_true',                    default=False)
    parser.add_argument('--ID_samples',     dest='num_samples',         type=int,   default=256)
    parser.add_argument('--error_samples',  dest='error_check_samples', type=int,   default=128)
    parser.add_argument('--threshold',      dest='error_threshold',     type=float, default=1e-3)
    parser.add_argument('--prop_threshold', dest='prop_threshold',      type=float, default=0.2)
    parser.add_argument('--epsilon',        dest='epsilon_for_pruning', type=float, default=1e-3)
    parser.add_argument('--max_it',         dest='max_it',              type=int,   default=4)
    parser.add_argument('--max_dofs',       dest='max_dofs',            type=int,   default=5000)
    parser.add_argument('--time_slice',     dest='time_slice',          type=float, default=1.0)
    parser.add_argument('--INR_key',        dest='INR_key',             type=str,   default='hos')
    parser.add_argument('--avg_error',      action='store_true',                    default=False)
    parser.add_argument('--paraview',       action='store_true',                    default=False)
    parser.add_argument('--num_uniform_ref', dest='num_uniform_ref', type=int, default=2, help='Number of uniform refinement iterations before adaptive refinement')
    args = parser.parse_args()
    

    return AMRConfig(
        inr_key             = args.INR_key,
        time_slice          = args.time_slice,
        max_it              = args.max_it,
        max_dofs            = args.max_dofs,
        num_samples         = args.num_samples,
        error_check_samples = args.error_check_samples,
        error_threshold     = args.error_threshold,
        prop_threshold      = args.prop_threshold,
        epsilon_for_pruning = args.epsilon_for_pruning,
        original            = args.original,
        avg_error           = args.avg_error,
        paraview            = args.paraview,
        num_uniform_ref     = args.num_uniform_ref
    )

def run_uniform_refinement(config: AMRConfig, mesh_handler: MeshHandler, inr, err_dof_dict, domain = None):
    """
    Run uniform refinement without error-based decision making.
    
    This function performs uniform mesh refinement where all elements are refined
    equally at each iteration. This is used for comparison with adaptive refinement
    and for initial uniform refinement phases. The function computes error metrics
    for each iteration to track the relationship between degrees of freedom and
    approximation accuracy.
    
    Args:
        config: AMR configuration object containing refinement parameters
        mesh_handler: Mesh handler object for mesh operations and updates
        inr: Pre-trained INR model for field evaluation
        err_dof_dict: Dictionary to store (error, DOF) pairs for each iteration
        domain: Cached domain information (optional, computed if None)
        
    Returns:
        dict: Updated err_dof_dict containing error and DOF information for each iteration
        
    Note:
        This function modifies the mesh_handler's mesh and updates grid functions
        to reflect the INR field values on the refined mesh.
    """
    for it in range(config.max_it):
        print(f"\nAMR iteration: {it}")
        print(f"Number of vtxs: {mesh_handler.fespace.GetNDofs()}")
        
        ne = mesh_handler.mesh.GetNE()
        local_error_v = mfem.Vector(ne)
        for i in range(ne):
            local_error_v[i] = 1
        
        mesh_handler.update_mesh(local_error_v)
        mesh_handler.save_mesh(it)

        # compute error
        if domain is None:
            # Get mesh vertices (in mesh coordinate frame)
            verts_mesh = mesh_handler.mesh.GetVertexArray()
            nv = len(verts_mesh)
            verts_mesh_flat = np.concatenate(verts_mesh, axis=0)
            verts_mesh_flat = verts_mesh_flat.reshape(nv, mesh_handler.dim)
            
            # Get processed vertices (in INR coordinate frame)
            verts_inr, _ = mesh_handler.get_processed_vertices()
            
            # Compute both domains
            mesh_domain = error.FindDomain(verts_mesh_flat, mesh_handler.dim)
            print("mesh domain = {}".format(mesh_domain))
            inr_domain  = error.FindDomain(verts_inr, mesh_handler.dim + ('CT' in config.inr_key)) # or 'PINN' in config.inr_key))
            print("INR domain  = {}".format(inr_domain))

            domain = {'mesh': mesh_domain, 'inr': inr_domain}
        rms_error = error.RMSE(
            mesh_handler.dim + ('CT' in config.inr_key ), #or 'PINN' in config.inr_key),
            domain,
            mesh_handler.mesh,  
            mesh_handler.gf_vtxs,
            inr,
            batch_size=2048,
            mesh_handler=mesh_handler
        )
        
        # recompute num dofs
        cdofs = mesh_handler.fespace.GetNDofs()
        err_dof_dict[it] = (rms_error.item(), cdofs)
        print(f"RMSE, DOFs = {err_dof_dict[it]}")

    return err_dof_dict

def run_adaptive_refinement(config: AMRConfig, mesh_handler: MeshHandler, inr, err_dof_dict, domain = None):
    """
    Run adaptive refinement using either PruningAMR or BasicAMR.
    
    This function implements the core PruningAMR algorithm, which uses neural network
    pruning to guide mesh refinement decisions. For each element, the algorithm:
    1. Applies Interpolative Decomposition (ID) pruning to the INR restricted to the element domain
    2. Computes error between original and pruned networks
    3. Checks if element needs refinement based on error and pruning thresholds
    4. Refines elements that exceed thresholds and updates mesh accordingly

    Alternatively, if using BasicAMR, the algorithm computes the average mean squared error over 
    the element between the mesh approximation and the original INR using `error_samples` samples.
    
    The algorithm uses parallel processing to compute errors for all elements simultaneously,
    making it efficient for large meshes.
    
    Args:
        config: AMR configuration object containing refinement parameters
        mesh_handler: Mesh handler object for mesh operations and updates
        inr: Pre-trained INR model for field evaluation
        err_dof_dict: Dictionary to store (error, DOF) pairs for each iteration
        domain: Cached domain information (optional, computed if None)
        
    Returns:
        dict: Updated err_dof_dict containing error and DOF information for each iteration
        
    Note:
        The algorithm automatically stops if maximum DOFs are reached or if no elements
        are refined in an iteration, indicating convergence.
    """
    num_workers = min(16, cpu_count())
    print(f"Performing Pruning AMR using {num_workers} workers")
    
    original_element_size = compute_original_element_size(mesh_handler)
    refinement_worker     = RefinementWorker(config, mesh_handler, inr, num_workers, original_element_size)
    
    prev_num_vtxs = 0
    
    for it in range(config.num_uniform_ref, config.max_it):
        cdofs = mesh_handler.fespace.GetNDofs()
        if cdofs >= config.max_dofs:
            print(f"==> Got {cdofs} dofs, which exceeded max threshold of {config.max_dofs}; quitting AMR loop")
            break
        elif cdofs == prev_num_vtxs and it > 1:
            print(f"==> Didn't refine anything after iteration {it}; quitting AMR loop")
            break
            
        print(f"\nAMR iteration: {it}")
        print(f"Starting number of vtxs: {cdofs}")
        prev_num_vtxs = cdofs
 
        # Compute local errors
        local_error = refinement_worker.compute_errors(it)
        local_error_v = mfem.Vector(len(local_error))
        for i in range(len(local_error)):
            local_error_v[i] = local_error[i]
            
        # Update and save mesh
        mesh_handler.update_mesh(local_error_v)
        mesh_handler.save_mesh(it)
        
        # Compute and save error metrics
        if domain is None:
            # Get mesh vertices (in mesh coordinate frame)
            verts_mesh = mesh_handler.mesh.GetVertexArray()
            nv = len(verts_mesh)
            verts_mesh_flat = np.concatenate(verts_mesh, axis=0)
            verts_mesh_flat = verts_mesh_flat.reshape(nv, mesh_handler.dim)
            
            # Get processed vertices (in INR coordinate frame)
            verts_inr, _ = mesh_handler.get_processed_vertices()
            
            # Compute both domains
            mesh_domain = error.FindDomain(verts_mesh_flat, mesh_handler.dim)
            inr_domain  = error.FindDomain(verts_inr, mesh_handler.dim + ('CT' in config.inr_key)) #or 'PINN' in config.inr_key))
            
            domain = {'mesh': mesh_domain, 'inr': inr_domain}

        rms_error = error.RMSE(
            mesh_handler.dim + ('CT' in config.inr_key), # or 'PINN' in config.inr_key),
            domain,
            mesh_handler.mesh,
            mesh_handler.gf_vtxs,
            inr,
            batch_size=2048,
            mesh_handler=mesh_handler
        )

        # recompute num dofs
        cdofs = mesh_handler.fespace.GetNDofs()
        err_dof_dict[it] = (rms_error.item(), cdofs)
        print(f"RMSE, DOFs = {err_dof_dict[it]}")
        
    return err_dof_dict

def save_parameters(config: AMRConfig, mesh_handler: MeshHandler, err_dof_dict: dict, domain):
    """
    Save parameters and results to JSON file for analysis and reproducibility.
    
    This function creates a comprehensive JSON file containing all configuration
    parameters, mesh information, error metrics, and domain information. This
    allows for easy analysis of results and reproduction of experiments.
    
    The saved parameters include:
    - Domain information (mesh and INR coordinate bounds)
    - Final mesh statistics (DOFs, initial vertices)
    - Error vs DOF progression for each iteration
    - Algorithm parameters (thresholds, sampling, etc.)
    
    Args:
        config: AMR configuration object containing all algorithm parameters
        mesh_handler: Mesh handler object providing mesh statistics
        err_dof_dict: Dictionary mapping iteration to (error, DOF) tuples
        domain: Cached domain information for coordinate transformations
        
    Note:
        The JSON file is saved as 'parameters.json' in the experiment directory
        and can be used for post-processing and analysis of results.
    """
    param_dict = {
        "domain": domain,
        "DOFs": mesh_handler.fespace.GetNDofs(),
        "init_num_vertices": mesh_handler.fespace.GetNDofs(),
        "err_dof_dict": err_dof_dict
    }
    
    if not config.original:
        param_dict.update({
            "num_samples": config.num_samples,
            "error_check_samples": config.error_check_samples,
            "error_threshold": config.error_threshold,
            "epsilon_for_pruning": config.epsilon_for_pruning
        })
    
    param_dict.update({
        "max_dofs": config.max_dofs,
        "max_it": config.max_it
    })
    
    with open(os.path.join(config.experiment_dir, "parameters.json"), "w") as outfile:
        json.dump(param_dict, outfile)

def main():
    """
    Main function to run the PruningAMR process.
    
    This function orchestrates the complete adaptive mesh refinement workflow:
    1. Parse command line arguments and create configuration
    2. Load the specified INR model from checkpoint
    3. Initialize mesh handler with appropriate mesh file
    4. Run either uniform or adaptive refinement based on configuration
    5. Save results including meshes, grid functions, and parameters
    
    The function supports two refinement modes:
    - Uniform refinement: Refines all elements equally (for comparison)
    - Adaptive refinement: Uses PruningAMR algorithm to selectively refine elements
      based on neural network pruning analysis or BasicAMR algorithm to selectively 
      refine elements based on the average mean squared error over the element between 
      the mesh approximation and the original INR.

    
    Output files are saved to the experiment directory with iteration-specific naming.
    """
    # Set up configuration
    config = parse_args()
    config.create_output_dirs()
    
    # Load INR model
    checkpoint_path = get_checkpoint_path(config.inr_key)
    inr = load_inr(config) #, checkpoint_path

    # Set up mesh handler
    mesh_handler = MeshHandler(config, get_mesh_file(config.inr_key))
    
    # Run refinement
    err_dof_dict = {}
    domain = None
    if config.original:
        run_uniform_refinement(config, mesh_handler, inr, err_dof_dict, domain)
    else:
        # Run num_uniform_ref iterations of uniform refinement first
        if config.num_uniform_ref > 0:
            orig_max_it = config.max_it
            # if config.max_it < config.num_uniform_ref, then run only config.max_it iterations of uniform refinement
            # otherwise, run config.num_uniform_ref iterations of uniform refinement
            if config.max_it >= config.num_uniform_ref:
                config.max_it = config.num_uniform_ref
            
            run_uniform_refinement(config, mesh_handler, inr, err_dof_dict, domain)
            config.max_it = orig_max_it
            print(f"==> Ran {config.num_uniform_ref} iterations of uniform refinement")
            print(f"==> Now running {config.max_it} iterations of adaptive refinement")
        
        # Now run adaptive refinement for the rest
        err_dof_dict = run_adaptive_refinement(config, mesh_handler, inr, err_dof_dict, domain)
    
    # Save parameters
    save_parameters(config, mesh_handler, err_dof_dict, domain)
    print(f"\n==> Final number of vertices: {mesh_handler.fespace.GetNDofs()}")

if __name__ == "__main__":
    main()
