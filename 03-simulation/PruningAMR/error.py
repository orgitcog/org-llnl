"""
Error computation module for PruningAMR algorithm.

This module provides error computation functions for both PruningAMR and BasicAMR
strategies. It includes domain computation, error estimation, and RMSE computation
for mesh refinement decisions.

Key functions:
- FindError(): PruningAMR error computation using ID pruning
- FindMeanError(): BasicAMR error computation using average error
- FindDomain(): Extract domain bounds from element vertices
- FindPropPruned(): Calculate proportion of pruned neurons
- RMSE(): Root mean square error computation for analysis of approximation accuracy
- RefineBools(): Convert error values to refinement decisions
- sample_domain(): Sample points uniformly from domain bounds
"""

import torch 
import math
import numpy as np
import mfem.ser as mfem
from mesh_handler import ConvertCoordinates, verify_NS_inr_coord
from typing import List, Union

def sample_domain(input_range: List[List[float]], 
                  k: int) -> torch.Tensor:
    """
    Sample points uniformly from a domain defined by input range.
    
    Generates k random samples from a hyperrectangular domain defined by
    the intervals in input_range. Each interval specifies [min, max] bounds
    for one dimension.
    
    Args:
        input_range: List of [min, max] intervals for each dimension
        k: Number of samples to generate
        
    Returns:
        torch.Tensor: Array of shape (k, len(input_range)) containing samples
    """
    output = np.zeros((k, len(input_range)))

    for i in range(len(input_range)):
        output[:, i] = np.random.uniform(low = input_range[i][0], high = input_range[i][1], size = k)

    return torch.tensor(output, dtype = torch.float32)

def FindDomain(vertices: np.ndarray, 
               dim: int) -> List[List[float]]:
        """
        Find domain bounds from element vertices.
        
        Extracts the bounding box of an element from its vertex coordinates.
        Returns min/max bounds for each dimension.
        
        Args:
            vertices: Array of vertex coordinates (shape: (2^dim, dim))
            dim: Dimension of the space (e.g., 3 for 3D)
            
        Returns:
            List of [min, max] bounds for each dimension
        """

        # initialize mins to -infinity and maxs to +infinity
        domain = []
        for component in range(dim):
                domain.append([math.inf, - math.inf])
        
        # update domain to match vertices
        for i in range(vertices.shape[0]):
                for d in range(dim):
                        vertex = vertices[i, :]
                        if vertex[d] < domain[d][0]:
                                domain[d][0] = vertex[d]
                        if vertex[d] > domain[d][1]:
                                domain[d][1] = vertex[d]

        return domain

def FindError(INR, 
              epsilon: float, 
              vertices: np.ndarray, 
              dim: int, 
              it: int, 
              original_element_size: float, 
              num_samples: int = 256, 
              error_check_samples: int = 512, 
              mesh_handler: MeshHandler = None) -> float:
        """
        Find PruningAMR error using ID pruning on element domain.
        
        Computes error by applying ID pruning to the INR on the element domain
        and comparing original vs pruned network outputs. Includes refinement
        history tracking to avoid wasted computation by re-refining elements 
        already determined to be done refining.
        
        Args:
            INR: Pre-trained INR model
            epsilon: Error tolerance for ID pruning
            vertices: Element vertex coordinates
            dim: Space dimension
            it: Current refinement iteration
            original_element_size: Size of original elements
            num_samples: Samples for ID pruning
            error_check_samples: Samples for error comparison
            mesh_handler: Mesh handler for coordinate transformations
            
        Returns:
            float: Relative error between original and pruned networks
        """

        # find domain
        domain = FindDomain(vertices, dim)
        # check if the first component (time for CT) is just a slice
        # if so, the mesh is one dimension smaller than the INR
        CT_flag   = False
        PINN_flag = False
        mesh_dim  = dim
        if domain[0][0] == domain[0][1]:
                mesh_dim = dim - 1;
                CT_flag  = True
        '''
        elif mesh_dim >=2:
                if domain[2][0] == domain[2][1]:
                    mesh_dim  = dim - 1
                    PINN_flag = True
        '''
        
        # find size of element
        if (dim == 2): # assuming dim 2 mesh has been refined twice <-- HACKY ***
                size = 8
        else:
                size = 1
        for i in range(dim):
                if CT_flag == True and i == 0:
                        pass
                elif PINN_flag == True and i==2:
                    pass 
                else:
                        size = size * (domain[i][1] - domain[i][0])
        # determine the last iteration this element was refined during
        num_uniform_ref = mesh_handler.config.num_uniform_ref
        last_refine_it = num_uniform_ref + int(np.emath.logn(2 ** mesh_dim, original_element_size / size))
        
        if (last_refine_it != it):
                # we have already determined that this element is done refining, so we can pass
                #print("last_refined_it=",last_refine_it, "it=",it,"; define error 0")

                # set INR.ks to list of zeros so that it won't be refined again
                INR.ks = np.zeros(INR.num_layers-1)

                # return 0 so that we don't refine again
                return 0 
        else:
                # prune network on domain
                INR.ID_prune(domain, epsilon = epsilon, num_samples = num_samples)
                # compare original and pruned networks to obtain error
                if isinstance(domain, dict):
                    # Use mesh domain for sampling (coordinates will be in mesh frame)
                    input_data = sample_domain(domain['mesh'], k = error_check_samples)
                else:
                    # Backward compatibility for old domain format
                    input_data = sample_domain(domain, k = error_check_samples)
                # Transform coordinates to INR frame if needed
                if mesh_handler and mesh_handler.config.inr_key == 'NS_PINN':
                    pass
                    input_data = ConvertCoordinates(input_data, to="INR")
                
                if mesh_handler.config.inr_key == 'NS_PINN':
                        verify_NS_inr_coord(input_data)
                with torch.inference_mode():
                        output_original = INR.forward(input_data, pruned = False) + 5
                        output_new      = INR.forward(input_data, pruned = True ) + 5
                # to avoid dividing by zero, threshold output_original to be >10^-3 in magnitude for division
                threshold = 10**-3
                #denominator = torch.nn.Threshold(threshold, threshold, inplace = False)(torch.abs(output_original))
                denominator = torch.abs(output_original)

                # Compute tensor relative difference
                abs_diff = torch.mean(torch.abs(output_original - output_new))
                rel_diff = torch.mean(torch.div(abs_diff, denominator))

                return rel_diff

def FindMeanError(INR: pretrained_INR, 
                  mesh: mfem.Mesh, 
                  gf: mfem.GridFunction, 
                  vertices: np.ndarray, 
                  dim: int, 
                  error_check_samples: int = 256, 
                  mesh_handler: MeshHandler = None) -> float:
        """
        Find BasicAMR error using average error between mesh and INR.
        
        Computes the average relative error between the mesh interpolant and
        the original INR on the element domain. This is used for BasicAMR
        refinement decisions.
        
        Args:
            INR: Pre-trained INR model
            mesh: MFEM mesh object
            gf: Grid function corresponding to mesh
            vertices: Element vertex coordinates
            dim: Space dimension
            error_check_samples: Number of samples for error computation
            mesh_handler: Mesh handler for coordinate transformations
            
        Returns:
            float: Average relative error between mesh and INR
        """

        # find domain
        domain = FindDomain(vertices, dim)

        # draw samples from domain
        if isinstance(domain, dict):
            # Use mesh domain for sampling (coordinates will be in mesh frame)
            input_data = sample_domain(domain['mesh'], k = error_check_samples)
        else:
            # Backward compatibility for old domain format
            input_data = sample_domain(domain, k = error_check_samples) # data in mesh frame

        # Create separate copies for INR and mesh operations
        inr_input_data  = input_data.clone()
        mesh_input_data = input_data.clone()

        # Transform coordinates to INR frame if needed
        if mesh_handler and mesh_handler.config.inr_key == 'NS_PINN':
            inr_input_data = ConvertCoordinates(inr_input_data, to="INR")


        # find output from mesh interpolant
        # first, find which elements the points belong to
        if dim == 4:
                mesh_input_data = mesh_input_data[:,1:]

        new_points, element_nums = _resample_missing_points(mesh, mesh_input_data, domain, dim) #mesh.FindPoints(mesh_input_data)

         # evaluate INR at samples
        if dim == 4:
                # add on time domain to new points in the first coordinate so that each point is (t, z, y, x)
                new_inr_points = np.concatenate((np.ones((new_points.shape[0], 1)) * domain[0][0], new_points), axis=1)
        else:
                new_inr_points = new_points

        # find true ouput from INR
        if mesh_handler.config.inr_key == 'NS_PINN':
                verify_NS_inr_coord(new_inr_points)
        with torch.inference_mode():
                output_INR = INR.forward(torch.tensor(new_inr_points, dtype = torch.float32), pruned = False)

        # then, compute interpolant output for each element num
        output_mesh = torch.zeros(output_INR.shape)        

        for i in range(error_check_samples):
                num  = element_nums[i]
                point = mfem.IntegrationPoint()
                if dim == 2:
                        x,y = np.float64(new_points[i,:].tolist())
                        point.Set2(x,y)
                else:
                        z,y,x = np.float64(new_points[i,:].tolist())
                        point.Set3(x,y,z)

                output_mesh[i,:] = gf.GetValue(num, point, dim-1)

        # compute mean squared error between the two
        denominator = torch.abs(output_INR)

        # Compute tensor relative difference
        abs_diff = torch.mean(torch.abs(output_INR - output_mesh))
        rel_diff = torch.mean(torch.div(abs_diff, denominator))

        return rel_diff


def FindPropPruned(INR: pretrained_INR) -> float:
        """
        Find proportion of pruned neurons in the pruned INR.
        
        Calculates the ratio of remaining neurons after pruning to the original
        number of neurons. 
        
        Args:
            INR: Pre-trained INR model with pruning information
            
        Returns:
            float: Proportion of remaining neurons, in (0, 1]
        """
        num_orig = 0
        num_new  = 0
        for i in range(INR.num_layers - 1):
            try:
                num_orig += INR.widths[i]
                num_new  += INR.ks[i]
            except:
                print("Error in computing num_new  += INR.ks[i]")
                print("Len(INR.ks) = {}".format(len(INR.ks)))

        return num_new/num_orig # this is the number of remaining hidden neurons (after pruning) divided by the original (non-pruned) number of hidden neurons


def RefineBools(errors: torch.Tensor, 
                error_threshold: float) -> int:
        """
        Convert error values to refinement decisions.
        
        Determines which elements should be refined based on error threshold.
        Returns 0 for elements below threshold (no refinement) and 1 for
        elements above threshold (refine).
        
        Args:
            errors: Array of error values for each element
            error_threshold: Maximum error allowed to stop refining
            
        Returns:
            int: 0 if error < threshold, 1 if error >= threshold
        """
        # check if components of errors are <= error_threshold
        return int(torch.ge(errors, error_threshold))


def RMSE(dim : int, 
         domain: List[List[float]], 
         mesh: mfem.Mesh, 
         gf: mfem.GridFunction, 
         INR: pretrained_INR, 
         batch_size: int = None, 
         mesh_handler: MeshHandler = None) -> float:
        """
        Compute Root Mean Square Error between mesh and INR.
        
        Calculates RMSE by sampling points from the domain and comparing
        mesh interpolant values with INR predictions. Used for analysis
        and convergence monitoring. The number of samples is determined 
        from the dimension of the mesh, and is given by (2 ** (M)) ** dim, 
        where M is a constant chosen based on the dimension of the mesh.
        
        Args:
            dim: Dimension of the mesh
            domain: Domain bounds for sampling
            mesh: MFEM mesh object
            gf: Grid function corresponding to mesh
            INR: Pre-trained INR model
            batch_size: Maximum points per batch (None for all at once)
            mesh_handler: Mesh handler for coordinate transformations
            
        Returns:
            float: Root mean square error between mesh and INR
        """
        
        # set M, to determine number of samples to take for RMSE computation
        M = 9
        if dim == 2:
                M = 9
        elif dim == 3:
                M = 7
        elif dim == 4:
                M = 5

        # compute number of samples
        num_samples = (2 ** (M)) ** dim

        # compute number of batches
        if batch_size:
                num_batches     = num_samples // batch_size + 1 # add one to account for last group of samples, which may be smaller than batch_size
                last_batch_size = num_samples % batch_size
        else:
                num_batches = 1 
                batch_size = num_samples

        # initialize running total for MSE
        sum_squares = 0
        for i in range(num_batches):
                if i == num_batches - 1:
                        batch_size_i = last_batch_size
                        if last_batch_size == 0:
                            continue
                else:
                        batch_size_i = batch_size

                # draw random samples
                if isinstance(domain, dict):
                    # Use INR domain for sampling (coordinates will be in INR frame)
                    inr_input_data  = sample_domain(domain['inr'], k = batch_size_i)
                    mesh_input_data = inr_input_data
                    #mesh_input_data = ConvertCoordinates(inr_input_data, to="mesh")
                else:
                    # Backward compatibility for old domain format
                    mesh_input_data = sample_domain(domain, k = batch_size_i)
                    # Transform coordinates to mesh frame if needed for mesh operations
                    #if mesh_handler and mesh_handler.config.inr_key == 'NS_PINN':
                    #    mesh_input_data = ConvertCoordinates(inr_input_data, to="mesh")

                # evaluate mesh at samples
                # find which elements the points belong to
                if dim == 4:
                    mesh_input_data = mesh_input_data[:,1:]

                # find element numbers; note that if some points are not found, we regenerate more points, which is why we also need to pass the new points too
                new_points, element_nums = _resample_missing_points(mesh, mesh_input_data, domain, dim) #mesh.FindPoints(mesh_input_data)

                # evaluate INR at samples
                if dim == 4:
                        # add on time domain to new points in the first coordinate so that each point is (t, z, y, x)
                        if isinstance(domain, dict):
                            # Use INR domain for time coordinate
                            time_coord = domain['inr'][0][0]
                        else:
                            # Backward compatibility for old domain format
                            time_coord = domain[0][0]
                        new_inr_points = np.concatenate((np.ones((new_points.shape[0], 1)) * time_coord, new_points), axis=1)
                else:
                       new_inr_points = new_points

                if mesh_handler.config.inr_key == 'NS_PINN':
                        verify_NS_inr_coord(new_inr_points)
                with torch.inference_mode():
                        output_INR = INR.forward(torch.tensor(new_inr_points, dtype = torch.float32), pruned = False)

                output_mesh = torch.zeros(output_INR.shape)                

                for i in range(batch_size_i):
                        num  = element_nums[i]
                        point = mfem.IntegrationPoint()
                        if dim == 2:
                                x,y = np.float64(new_points[i,:].tolist())
                                point.Set2(x,y)
                        elif dim == 3:
                                x, y, t = np.float64(new_points[i,:].tolist())
                                point.Set3(x,y,t)
                        else:
                                z,y,x = np.float64(new_points[i,:].tolist())
                                point.Set3(x,y,z)
                        if num != -1:
                            output_mesh[i,:] = gf.GetValue(num, point, dim-1)
                            #print(output_mesh[i,:])
                        else:
                            print("Point not found: i=",i,"num=",num,"x=",x,"y=",y)
                            if dim == 3: print("t = {}".format(t))
                            print("num = {}".format(num))
                            print("Removing point from sample")
                            # remove point from sample by setting corresponding point in output_INR and in output_mesh to 0
                            exit()
                            # output_mesh is already set to 0 for this point, so only need to update output_INR
                            output_INR[i,:] = torch.zeros(output_INR[i,:].shape)

                # compute squared error
                sum_squares += torch.sum(torch.square(output_INR - output_mesh)) / num_samples

        return torch.sqrt(sum_squares)

import numpy as np

def _resample_missing_points(mesh, pts_np, domain, dim, max_resamples=50, seed=42):
    """
    Try locating points with mesh.FindPoints; for any that return -1 (not found),
    draw *exactly that many* replacement points uniformly from the given domain
    and try again. Repeat up to `max_resamples` times.

    Parameters
    ----------
    mesh : mfem.Mesh
        The MFEM mesh to query.
    pts_np : np.ndarray, shape (npts, dim), dtype float64
        Physical coordinates of the points to locate. This array may be modified
        in-place for the rows corresponding to points that are replaced.
    domain : Sequence[Sequence[float]]
        Global domain bounds per coordinate: [[x_lo, x_hi], [y_lo, y_hi], ...].
        Assumed to be correct (no recomputation is performed).
    dim : int
        Spatial dimension (typically mesh.Dimension()).
    max_resamples : int, optional
        Maximum number of replacement attempts for still-missing points.
    seed : Optional[int], optional
        Seed for reproducible random replacement point generation.

    Returns
    -------
    pts_np_updated : np.ndarray, shape (npts, dim)
        The (possibly updated) coordinates where any missing points were replaced.
    elem_ids_np : np.ndarray, shape (npts,), dtype int64
        Element ids for each point (>= 0 if found, -1 if still missing after retries).

    Notes
    -----
    - MFEM expects a (dim, npts) array for FindPoints; we pass pts_np.T.
    - Only points that fail are replaced; successful points are left untouched.
    - If some points are still not found after `max_resamples`, their elem ids
      remain -1 in the returned array.
    """
    rng = np.random.default_rng(seed)

    # Initial locate on the provided points
    # elem = mesh.FindPoints(pts_np)  # MFEM expects (dim, npts)
    # wrap FindPoints so that warning messages don't come through to terminal
    out = getattr(mfem, "out") if hasattr(mfem, "out") else mfem.out()
    try:
        # some builds expose .Disable directly on the object, some via call
        (out.Disable() if hasattr(out, "Disable") else out().Disable())
        elem =  mesh.FindPoints(pts_np)
    finally:
        (out.Enable() if hasattr(out, "Enable") else out().Enable())

    n = len(elem[1])

    # Convert MFEM array to numpy array - handle IntegrationPoint objects properly
    elem_ids_np = np.zeros(n, dtype=np.int64)
    for i in range(n):
        elem_ids_np[i] = elem[1][i]

    # Indices of points that were not found (-1)
    missing = np.flatnonzero(elem_ids_np < 0)

    tries = 0
    while missing.size > 0 and tries < max_resamples:
        nmiss = missing.size
        print("Replacing {} missing points".format(nmiss))

        # Draw replacement points uniformly within the global domain bounds
        if isinstance(domain, dict):
            # Use mesh domain for sampling (coordinates will be in mesh frame)
            replacement_points = sample_domain(domain['mesh'], k = nmiss)
        else:
            # Backward compatibility for old domain format
            replacement_points = sample_domain(domain, k = nmiss)

        # Test only the replacement points
        #elem_repl = mesh.FindPoints(replacement_points)


        # wrap FindPoints so that warning messages don't come through to terminal
        out = getattr(mfem, "out") if hasattr(mfem, "out") else mfem.out()
        try:
            # some builds expose .Disable directly on the object, some via call
            (out.Disable() if hasattr(out, "Disable") else out().Disable())
            elem_repl =  mesh.FindPoints(replacement_points)
        finally:
            (out.Enable() if hasattr(out, "Enable") else out().Enable())

        elem_repl_np = np.zeros(nmiss, dtype=np.int64)
        for i in range(nmiss):
            elem_repl_np[i] = elem_repl[1][i]

        # Keep successful replacements; leave failures for another iteration
        found_mask = elem_repl_np >= 0
        if np.any(found_mask):
            target_rows = missing[found_mask]
            pts_np[target_rows, :] = replacement_points[found_mask, :]
            elem_ids_np[target_rows] = elem_repl_np[found_mask]

        # Update the set of indices still missing and loop again if needed
        missing = np.flatnonzero(elem_ids_np < 0)
        tries += 1

    return pts_np, elem_ids_np





