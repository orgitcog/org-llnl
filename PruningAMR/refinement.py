"""
Parallel error computation and mesh refinement module for PruningAMR.

This module provides parallel computation of local errors for mesh refinement
decisions in the PruningAMR algorithm. It supports both PruningAMR and BasicAMR
error computation strategies with multi-threaded processing.

Key components:
- RefinementWorker: Main class for parallel error computation
- compute_original_element_size(): Calculate original element size for refinement tracking
- Multi-threaded error computation with progress tracking
- Support for both pruning-based and average error strategies
"""

import  numpy    as np
import  torch
import  threading
from    queue    import Queue
from    tqdm     import tqdm
import  copy
from    typing   import Tuple, List, Optional

import  error
from    config          import AMRConfig
from    mesh_handler    import MeshHandler
from    ID_pruning      import pretrained_INR

class RefinementWorker:
    """
    Class to handle parallel refinement computations.
    
    Manages multi-threaded error computation for mesh refinement decisions.
    Supports both PruningAMR (ID-based) and BasicAMR (average error) strategies
    with configurable number of worker threads for efficient parallel processing.
    """
    
    def __init__(self, config: AMRConfig, mesh_handler: MeshHandler, inr: pretrained_INR, 
                 num_workers: int, original_element_size: float):
        """
        Initialize the refinement worker.
        
        Sets up the worker with configuration, mesh handler, INR model, and
        parallel processing parameters for error computation.
        
        Args:
            config: AMR configuration object
            mesh_handler: Mesh handler object for mesh operations
            inr: INR model for error computation
            num_workers: Number of worker threads for parallel processing
            original_element_size: Size of original elements for refinement tracking
        """
        self.config                = config                 # AMR configuration object
        self.mesh_handler          = mesh_handler           # Mesh handler object
        self.inr                   = inr                    # INR model
        self.num_workers           = num_workers            # Number of worker threads
        self.original_element_size = original_element_size  # Size of original elements
        
    def compute_local_error(self, i: int, local_INR: pretrained_INR, current_it: int) -> float:
        """
        Compute local error for an element.
        
        Computes error for a single mesh element using either PruningAMR or BasicAMR
        strategy based on configuration. For PruningAMR, applies ID pruning and
        checks both error and proportion thresholds. For BasicAMR, computes average
        error between mesh and INR.
        
        Args:
            i: Element index
            local_INR: Local copy of INR model for thread safety
            current_it: Current iteration number
            
        Returns:
            float: Error value for the element (0 or 1 for refinement decision)
        """
        torch.set_grad_enabled(False)
        el = self.mesh_handler.mesh.GetElement(i)
        coords = self.mesh_handler.get_processed_vertices()[0][el.GetVerticesArray(),:] # coordinates are in mesh frame
        if self.config.avg_error:
            if ('CT' in self.config.inr_key): #' in self.config.inr_key):
                error_i = error.FindMeanError(
                    local_INR, 
                    self.mesh_handler.mesh, 
                    self.mesh_handler.gf_vtxs, 
                    coords, 
                    self.mesh_handler.dim + 1, 
                    self.config.error_check_samples,
                    self.mesh_handler
                )
            else:
                error_i = error.FindMeanError(
                    local_INR, 
                    self.mesh_handler.mesh, 
                    self.mesh_handler.gf_vtxs, 
                    coords, 
                    self.mesh_handler.dim, 
                    self.config.error_check_samples,
                    self.mesh_handler
                )
            return float(error.RefineBools(error_i, self.config.error_threshold))
            
            
        if ('CT' in self.config.inr_key): # or ('PINN' in self.config.inr_key):
            error_i = error.FindError(
                local_INR, 
                self.config.epsilon_for_pruning, 
                coords, 
                self.mesh_handler.dim + 1, 
                current_it, 
                self.original_element_size,
                self.config.num_samples, 
                self.config.error_check_samples,
                self.mesh_handler
            )
        else:
            error_i = error.FindError(
                local_INR, 
                self.config.epsilon_for_pruning, 
                coords, 
                self.mesh_handler.dim, 
                current_it, 
                self.original_element_size,
                self.config.num_samples, 
                self.config.error_check_samples,
                self.mesh_handler
            )
        proportion = error.FindPropPruned(local_INR)
        return float(error_i > self.config.error_threshold or proportion > self.config.prop_threshold)
    
    def worker(self, queue: Queue, model: pretrained_INR, local_error: np.ndarray, 
               pbar: tqdm, pbar_lock: threading.Lock, current_it: int) -> None:
        """
        Worker function for parallel processing.
        
        Processes elements from the task queue in parallel, computing local errors
        and updating the shared error array. Includes thread-safe progress tracking.
        
        Args:
            queue: Task queue containing element indices
            model: Local copy of INR model for thread safety
            local_error: Shared array to store error values
            pbar: Progress bar for tracking completion
            pbar_lock: Thread lock for progress bar updates
            current_it: Current iteration number
        """
        while True:
            i = queue.get()
            if i is None:
                queue.task_done()
                break
            local_error[i] = self.compute_local_error(i, model, current_it)
            with pbar_lock:
                pbar.update(1)
            queue.task_done()
    
    def compute_errors(self, current_it: int) -> np.ndarray:
        """
        Compute errors for all elements in parallel.
        
        Distributes element error computation across multiple worker threads
        for efficient parallel processing. Uses task queues and thread-safe
        progress tracking.
        
        Args:
            current_it: Current iteration number
            
        Returns:
            np.ndarray: Array of error values for all elements
        """
        ne = self.mesh_handler.mesh.GetNE()
        queues = [Queue() for _ in range(self.num_workers)]
        local_error = np.empty(ne)
        
        # Create a thread-safe progress bar
        pbar = tqdm(total=ne, desc="Processing elements", position=0, leave=True)
        pbar_lock = threading.Lock()
        
        # Start worker threads
        threads = []
        for q in queues:
            t = threading.Thread(
                target=self.worker,
                args=(q, copy.deepcopy(self.inr), local_error, pbar, pbar_lock, current_it)
            )
            t.start()
            threads.append(t)
        
        # Distribute work
        for i in range(ne):
            queues[i % self.num_workers].put(i)
        
        # Add poison pills and wait for completion
        for q in queues:
            q.put(None)
        for t in threads:
            t.join()
        
        pbar.close()
        return local_error

def compute_original_element_size(mesh_handler: MeshHandler) -> float:
    """
    Compute the size of original elements.
    
    Calculates the volume of elements in the original mesh for tracking
    refinement history. Handles different INR types with appropriate dimension
    calculations.
    
    Args:
        mesh_handler: Mesh handler object containing mesh information
        
    Returns:
        float: Size of original elements for refinement tracking
    """
    el_1   = mesh_handler.mesh.GetElement(0)
    coords = mesh_handler.get_processed_vertices(INR_frame = False)[0][el_1.GetVerticesArray(), :]
    
    if "CT" in mesh_handler.config.inr_key:
        el_domain = error.FindDomain(coords, mesh_handler.dim + 1)
        size = 1.0
        for i in range(mesh_handler.dim + 1):
            if i != 0:
                size *= (el_domain[i][1] - el_domain[i][0])
        return size
        '''
        elif "PINN" in mesh_handler.config.inr_key:
        el_domain = error.FindDomain(coords, mesh_handler.dim + 1)
        size = 1.0
        for i in range(mesh_handler.dim + 1):
            if i != 2:
                size *= (el_domain[i][1] - el_domain[i][0])
        return size
        '''
    else:
        el_domain = error.FindDomain(coords, mesh_handler.dim)
        size = 1.0
        for i in range(mesh_handler.dim):
            size *= (el_domain[i][1] - el_domain[i][0])
        return size 
