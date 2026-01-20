# risingMicrobubbleLattice

This code is a simulation case to be run with OpenFOAM, an open-source computational fluid dynamics software. A gmsh mesh file is also included. Specifically, this simulation demonstrates the transport of a single microbubble rising through an ordered lattice due to an applied flow. The bubble deforms as it squeezes through the pores of the lattice. This code accompanies the manuscript in review:

Guo, J., Seung, K., Kang, S., Lin, T.Y., Davis, J.T. Bubble transport through a porous lattice with an applied inlet flow. Manuscript in prep. 

Below are some additional notes on running the computational case corresponding to Case L, with the interFoam solver. For more detailed instructions, follow the guidelines in OpenFOAM's "Breaking of a dam" tutorial.

Notes:
1. An empty "foam.foam" file is recommended in order to view the results in paraview.
2. This directory needs to be set up to run with the D = 300 micron mesh; to set up the directory to be able to run with a mesh:
    1. Copy the 0.orig/ directory to a new directory (named 0/).
    2. For a given "mesh_file.geo" mesh file, use "gmsh -3 mesh_file.geo" to create the mesh_file.msh file. Bring the mesh_file.msh file into the desired run directory.
    3. Run "gmshToFoam mesh_file.msh". This will change and create certain files within the directory to incorporate the information of the mesh. See OpenFOAM documentation for more details.
    4. Then, in constant/polymesh/boundary, for walls with symmetry boundary conditions, change:  
        type            patch;  
to  
        type            symmetry;  
and also comment out:  
        physicalType    patch;  
  
3. To change the bubble size or initial condition, modify system/setFieldsDict and then run the "setFields" command. See OpenFOAM documentation for more details.

LLNL-CODE-2007876
