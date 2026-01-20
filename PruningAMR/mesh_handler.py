"""
Mesh handling module for PruningAMR algorithm.

This module provides functionality for mesh operations, vertex processing, and
visualization in the PruningAMR system. It handles different INR types with
appropriate coordinate transformations and mesh refinement operations.

Key components:
- MeshHandler: Main class for mesh operations and refinement
- Vertex processing functions for different INR types (CT, PINN)
- Coordinate transformation utilities
- Mesh saving and ParaView visualization support

The module integrates with MFEM for mesh operations and supports both
uniform and adaptive refinement strategies.
"""

import numpy as np
import torch
from typing import Tuple, Optional
import mfem.ser as mfem
from os.path import dirname
import os

from config import AMRConfig
from ID_pruning import pretrained_INR
import inr_setup

def process_4D(vertices: np.ndarray, nv: int, t: float) -> np.ndarray:
    """
    Process 4D CT vertices for 3D visualization of time slices.
    
    Transforms 3D mesh vertices to 4D coordinates by adding time dimension
    and reordering coordinates for CT INR compatibility.
    
    Args:
        vertices: Array of 3D vertex coordinates
        nv: Number of vertices
        t: Time slice value to add as first dimension
        
    Returns:
        Processed 4D vertices array with time dimension
    """
    vertices[:,[0,2]] = vertices[:,[2,0]]
    vertices = np.insert(vertices, 0, t * np.ones(nv), axis=1)
    return vertices

def process_PINN(vertices: np.ndarray, nv: int, t: float) -> np.ndarray:
    """
    Process PINN vertices.
    
    Currently returns vertices unchanged. Placeholder for future PINN-specific
    vertex processing if needed.
    
    Args:
        vertices: Array of vertex coordinates
        nv: Number of vertices
        t: Time slice value
        
    Returns:
        Processed vertices array (currently unchanged)
    """
    #vertices = np.insert(vertices, 2, t * np.ones(nv), axis=1)
    return vertices

def process_NS_PINN(vertices: np.ndarray, nv: int, t: float, INR_frame : bool = True) -> np.ndarray:
    """
    Process NS_PINN vertices with coordinate transformation.
    
    Handles coordinate transformation for NS-PINN models which have different
    reference frames between mesh and INR domains. Currently returns vertices
    unchanged but includes transformation logic for future use.
    
    Args:
        vertices: Array of vertex coordinates (from [0,4] × [0,4] mesh)
        nv: Number of vertices
        t: Time slice value
        INR_frame: Whether to apply INR coordinate transformation
        
    Returns:
        Processed vertices array (currently unchanged)
    """
    # Transform coordinates: x -> x+1, y -> y-2
    if False and INR_frame:
        vertices = ConvertCoordinates(vertices, to = "INR")
    
    # Add time dimension
    #vertices = np.insert(vertices, 2, t * np.ones(nv), axis=1)
    return vertices

def ConvertCoordinates(vertices: np.ndarray, to = "INR") -> np.ndarray:
    """
    Convert between INR and mesh coordinate frames.
    
    Designed for NS-PINN models which have different reference frames between
    mesh coordinates and INR domain. Currently returns vertices unchanged.
    
    Args:
        vertices: Array of vertex coordinates
        to: String, either "INR" or "mesh", denoting target coordinate frame
        
    Returns:
        Converted vertices array (currently unchanged)
    """
    return vertices
    v = vertices.clone() if torch.is_tensor(vertices) else vertices.copy() 

    if to == "INR":
        # Transform coordinates: x -> x+1, y -> y-2
        v[:, 0] = v[:, 0] + 1  # x coordinate: [0,4] -> [1,5]
        v[:, 1] = v[:, 1] - 2  # y coordinate: [0,4] -> [-2,2]
    elif to == "mesh":
        v[:, 0] = v[:, 0] - 1  # x coordinate: [1, 5]  -> [0,4]
        v[:, 1] = v[:, 1] + 2  # y coordinate: [-2, 2] -> [0,4]
    else:
        print("Error: argument 'to' must be either 'INR' or 'mesh', corresponding to the desired reference frame to swtich to")
        exit()
    return v

def verify_NS_inr_coord(inr_pts):
    """
    Check that vertices are in the domain of the NS PINN INR.
    
    Currently returns without checking. Placeholder for domain validation
    of NS-PINN coordinate ranges.
    """
    return
    x = inr_pts[..., 0]        # works for 1-D or N×3
    y = inr_pts[..., 1]

    assert (x >= 1).all() and (x <= 5).all() and (y >= -2).all() and (y <= 2).all()

class MeshHandler:
    """
    Main class for handling mesh operations in PruningAMR.
    
    This class manages mesh initialization, refinement, vertex processing, and
    paraview saving for the PruningAMR algorithm. It integrates with MFEM for
    mesh operations and handles different INR types with appropriate coordinate
    transformations.
    
    Key capabilities:
    - Mesh initialization and finite element space setup
    - Vertex processing for different INR types
    - Mesh refinement based on error vectors
    - Grid function management and INR field evaluation
    - Mesh saving and ParaView saving
    """
    
    def __init__(self, config: AMRConfig, meshfile: str):
        """
        Initialize the mesh handler.
        
        Sets up mesh, finite element space, grid functions, and loads the INR model.
        Initializes grid function values if using average error computation.
        
        Args:
            config: AMR configuration object
            meshfile: Path to the mesh file
        """
        self.config = config
        self.order  = 1                           # Order of the finite element space
        self.mesh   = mfem.Mesh(meshfile, 1, 1)
        self.dim    = self.mesh.Dimension()       # Dimension of the mesh
        self.sdim   = self.mesh.SpaceDimension()  # Space dimension of the mesh
        
        # Set up finite element space
        self.fec     = mfem.H1_FECollection(self.order, self.dim)
        self.fespace = mfem.FiniteElementSpace(self.mesh, self.fec)
        
        # Set up grid function
        self.gf_vtxs_fec = mfem.H1_FECollection(self.order, self.dim)
        self.gf_vtxs_fes = mfem.FiniteElementSpace(self.mesh, self.gf_vtxs_fec)
        self.gf_vtxs     = mfem.GridFunction(self.gf_vtxs_fes)
        
        # set up INR
        self.inr = inr_setup.load_inr(config)
        print("set up INR in mesh handler")
 
        if self.config.avg_error:
            self.init_gridfunction_values()

    def get_processed_vertices(self, INR_frame = True) -> Tuple[np.ndarray, int]:
        """
        Get processed vertices based on INR type.
        
        Applies appropriate coordinate transformations based on the INR model type:
        - CT models: Adds time dimension and reorders coordinates
        - NS_PINN: Applies coordinate transformations for different reference frames (currently disabled)
        - PINN models: Returns vertices unchanged
        
        Args:
            INR_frame: Whether to apply INR coordinate transformations
            
        Returns:
            Tuple containing processed vertices array and number of vertices
        """
        verts = self.mesh.GetVertexArray()
        nv = len(verts)
        verts2 = np.concatenate(verts, axis=0)
        verts2 = verts2.reshape(nv, self.dim)
        
        if 'CT' in self.config.inr_key:
            verts2 = process_4D(verts2, nv, self.config.time_slice)
        elif self.config.inr_key == 'NS_PINN':
            verts2 = process_NS_PINN(verts2, nv, self.config.time_slice, INR_frame = INR_frame)
        elif 'PINN' in self.config.inr_key:
            verts2 = process_PINN(verts2, nv, self.config.time_slice)
            
        return verts2, nv
    
    def init_gridfunction_values(self) -> None:
        """
        Initialize grid function values for average error computation.
        
        Evaluates the INR at all mesh vertices and stores the values in the
        grid function. This is required for BasicAMR error computation.
        """
        self.gf_vtxs_fes.Update()
        self.gf_vtxs.Update()
        
        verts2, nv = self.get_processed_vertices()
        for j in range(nv):
            vtx_to_pass = torch.tensor(verts2[j,:]).float()
            if self.config.inr_key == 'NS_PINN':
                verify_NS_inr_coord(vtx_to_pass)
            with torch.inference_mode():
                inr_val = self.inr.forward(vtx_to_pass)
            try:
                self.gf_vtxs[j] = float(inr_val.squeeze()[0].detach())
            except:
                self.gf_vtxs[j] = float(inr_val.squeeze().item())
    
    def update_mesh(self, local_error_v: mfem.Vector) -> None:
        """
        Update mesh based on error vector.
        
        Refines mesh elements based on local error values and updates all
        associated finite element spaces and grid functions. Re-evaluates
        INR values at new vertices.
        
        Args:
            local_error_v: Vector of local error values for each element
        """
        self.mesh.RefineByError(local_error_v, 0.5, -1, 0)
        self.fespace.Update()
        
        # Update grid functions
        self.gf_vtxs_fes.Update()
        self.gf_vtxs.Update()
        
        # Update vertex values
        verts2, nv = self.get_processed_vertices()
        for j in range(nv):
            vtx_to_pass = torch.tensor(verts2[j,:]).float()
            if self.config.inr_key == 'NS_PINN':
                verify_NS_inr_coord(vtx_to_pass)
            with torch.inference_mode():
                inr_val = self.inr.forward(vtx_to_pass)
            try:
                self.gf_vtxs[j] = float(inr_val.squeeze()[0].detach())
            except:
                self.gf_vtxs[j] = float(inr_val.squeeze().item())
    
    def save_mesh(self, iteration: int) -> None:
        """
        Save mesh and grid function for current iteration.
        
        Saves the refined mesh and associated grid function values to files.
        Optionally generates ParaView output if enabled in configuration.
        
        Args:
            iteration: Current iteration number for file naming
        """
        # Fix continuity issue for grid function
        gf2save = self.gf_vtxs
        gf2save.SetTrueVector()
        gf2save.SetFromTrueVector()
        
        # Save mesh and grid function
        self.mesh.Save(f"{self.config.experiment_dir}/it_{iteration}.mesh")
        gf2save.Save(f"{self.config.experiment_dir}/it_{iteration}.gf", 8)
        print(f"Saved output to {self.config.experiment_dir}/it_{iteration} + .mesh and .gf")
        
        if self.config.paraview:
            self.save_paraview(gf2save)
    
    def save_paraview(self, gf2save: mfem.GridFunction) -> None:
        """
        Save mesh in ParaView format.
        
        Creates ParaView-compatible files for visualization, including
        mesh geometry and grid function data. Files are saved in a
        ParaView subdirectory within the experiment directory.
        
        Args:
            gf2save: Grid function to save with the mesh
        """
        pv = mfem.ParaViewDataCollection(self.config.experiment_dir, self.mesh)
        paraview_dir = f"{self.config.experiment_dir}/ParaView"
        os.makedirs(paraview_dir, exist_ok=True)
        print(f"New directory made at: {paraview_dir}")
        
        pv.SetPrefixPath(paraview_dir)
        pv.SetHighOrderOutput(False)
        pv.SetLevelsOfDetail(self.order)
        pv.RegisterField("INRgf", gf2save)
        pv.SetCycle(0)
        pv.Save()
        print("==> Saved Paraview format to ParaView subfolder") 
