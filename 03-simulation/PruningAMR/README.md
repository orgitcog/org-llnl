# PruningAMR: Efficient Visualization of Implicit Neural Representations via Weight Matrix Analysis

This repository implements PruningAMR, an algorithm for efficient visualization of implicit neural representations (INRs) by using techniques from neural network compression and adaptive mesh refinement (AMR). See our preprint "Pruning AMR: Efficient Visualization of Implicit Neural Representations via Weight Matrix Analysis" for more information on the algorithm and examples.

## Overview

The PruningAMR algorithm combines:
- **Implicit Neural Representations (INRs)** for continuous field representation
- **Adaptive Mesh Refinement (AMR)** for building variable resolution
- **Neural Network Pruning** using Interpolative Decomposition (ID) to guide refinement decisions

## Key Features

- Support for different INR architectures
- Parallel error computation using multi-threading
- Integration with [MFEM](https://mfem.org/) for mesh refinement operations
- Ability to save to files compatible with visualization in Parview
- Option to view specific time slices of 4D (3D space + time) INRs

### Requirements

- python 3.9.12
- torch 2.3.0
- [PyMFEM](https://github.com/mfem/PyMFEM) 4.6.1.0
- numpy 1.26.4
- tqdm 4.66.4
- scipy 1.13.0

## Usage

### Basic Command

```bash
python main.py --INR_key <model_key> [options]
```

### Command Line Arguments

| Argument            | Type  | Default | Description                                                                                                     |
| ------------------- | ----- | ------- | --------------------------------------------------------------------------------------------------------------- |
| `--original`        | flag  | False   | Run uniform refinement instead of adaptive                                                                      |
| `--ID_samples`      | int   | 256     | Number of samples for ID pruning                                                                                |
| `--error_samples`   | int   | 128     | Number of samples for error checking                                                                            |
| `--threshold`       | float | 1e-3    | Error threshold for refinement                                                                                  |
| `--prop_threshold`  | float | 0.2     | Proportion threshold for pruning                                                                                |
| `--epsilon`         | float | 1e-3    | Epsilon parameter for pruning                                                                                   |
| `--max_it`          | int   | 4       | Maximum number of refinement iterations                                                                         |
| `--max_dofs`        | int   | 5000    | Maximum number of degrees of freedom                                                                            |
| `--time_slice`      | float | 1.0     | Time slice for 4D data (e.g. CT scans)                                                                          |
| `--INR_key`         | str   | 'hos'   | INR model identifier                                                                                            |
| `--avg_error`       | flag  | False   | Use BasicAMR instead of PruningAMR                                                                              |
| `--paraview`        | flag  | False   | Generate ParaView output files                                                                                  |
| `--num_uniform_ref` | int   | 2       | Number of uniform refinement iterations before adaptive refinement begins; counts towards total iteration count |

### Available INR Models

The system supports several pre-trained INR models. To load your own pre-trained network, simply save and then load (set a path to your model in `config.py`) a dictionary named `save_dict` with the following:
- `save_dict[weights_list]` set to a list of weight tensors for your network, in order.
- `save_dict[bias_list]` set to a list of bias tensors for your network, in order.
- `save_dict[act_list]` set to a list of activation functions for your network, in order. Note that the output layer should be included (can use the identity function if no activation is applied).
- **optional**: Save any additional variables needed for transformations on the network's inputs or outputs

The pre-trained INRs included in this repository are:
#### Oscillatory 2D Function Models
- `hos` - High oscillation small (2D model in paper)
- `hol` - High oscillation long
- `hof` - High oscillation full

#### PINN Models
- `NS_PINN` - Navier-Stokes PINN for flow around a circular cylinder; trained by adapting code from [this repo](https://github.com/Shengfeng233/PINN-for-NS-equation). 

#### CT Scan Models
The CT scan INR code and weights are available through a licensing process. Please contact the authors of the paper [Distributed Stochastic Optimization of a Neural Representation Network for Time-Space Tomography Reconstruction](https://ieeexplore.ieee.org/abstract/document/10908803?casa_token=_2jNLHPK9coAAAAA:I03Y762W1CKKCpQYMrlYXHt-2QAVVY_qUjNHo2JJ7bISzdM3Gq8OEAYVfAgCVO76egPkV4fg) if you are interested. 


## File Structure

### Core Modules

- **`main.py`** - Main script for preparing the mesh, orchestrating refinement, reporting accuracy, and saving mesh files.
- **`config.py`** - Configuration management and INR model definitions
- **`inr_setup.py`** - Loading and initialization of pre-trained INR models
- **`mesh_handler.py`** - Mesh operations and vertex processing: initialization, updating, and saving of mesh
- **`refinement.py`** - Set up refinement workers for error computation in parallel
- **`error.py`** - Error estimation and domain analysis
- **`ID_pruning.py`** - Network class that initializes from weights, bias, and activation function lists; functionality for pruning INR using Interpolative Decomposition. 

### Supporting Files

- **`utils.py`** - Utility function for making directories
- **`ID_Pruning/ID.py`** - Utilities for computing Interpolative Decompositions

### Data Directories
- **`checkpoints/`** - Pre-trained INR model checkpoints
- **`meshes/`** - Source mesh files for different INR model types
- **`output/`** - Generated mesh and visualization files

#### Mesh Files
The `meshes/` directory contains source mesh files used for different INR model types:
- **`box-hex.mesh`** - 3D hexahedral mesh for CT scan models
- **`NS_PINN_2x2x10_z0_20.mesh`** - 3D mesh for NS-PINN flow around cylinder
- **`NS_PINN_2x2x21.mesh`** - Alternative 3D mesh for NS-PINN
- **`NS_PINN.mesh`** - Basic 3D mesh for NS-PINN
- **`PINN.mesh`** - 2D mesh for PINN models
- **`rectangle.mesh`** - 2D rectangular mesh for oscillatory function models

The appropriate mesh file is automatically selected based on the `--INR_key` parameter.

## Algorithm Overview

### 1. Initialization
- Load the specified INR model (designated using `--INR_key` command line argument) from pre-trained weights, biases, and activation functions
- Initialize mesh corresponding to that INR model
- Set up finite element space and grid functions, which will be used to approximate the INR on the mesh

### 2. Uniform Refinement Phase (Optional)
- Perform `num_uniform_ref` iterations of uniform refinement
    - If the command line argument `--original` is used, then all of the iterations will be uniform refinement
- Compute error metrics for each iteration

### 3. Adaptive Refinement Phase
- For each element in the mesh:
    - if `avg_error` is False:
        - Apply ID pruning to the INR restricted to the element domain (amount of pruning is dependent on `epsilon` and `ID_samples`)
        - Compute error between original and pruned networks on that element (number of samples used to compute error is given by `error_samples`)
        - Check if element needs refinement based on error and pruning thresholds (`threshold` and `prop_threshold`, respectively)
    - if `avg_error` is True:
        - Compute average mean squared error over the element between the mesh approximation and the original INR using `error_samples` samples
        - Check if element needs refinement based on error threshold (`threshold`)
- Refine elements that exceed either threshold
- Update mesh and grid functions
- Repeat until convergence, maximum iterations (`max_it`), or maximum degrees of freedom (`max_dofs`) are reached 

### 4. Output Generation
- Save refined meshes and grid functions
- Generate ParaView files (if requested by command line argument `paraview`)
- Save parameters and results to JSON

## Key Parameters

### Error Threshold (`--threshold`)
- Controls sensitivity of refinement decisions
- Smaller values → more aggressive refinement
- With `--avg_error`: refine if local error > threshold
- Without `--avg_error`: refine if pruned network error > threshold

### Proportion Threshold (`--prop_threshold`)
- Only use if 'avg_error' and 'original' are both False (this is the default choice)
- Refine if ratio of pruned to original neurons > threshold
- Smaller values → more refinement (demands accurate representation with smaller network)

### Epsilon (`--epsilon`)
- Controls pruning aggressiveness in ID algorithm
- Smaller values → less pruning → more refinement

## Output Files

The system generates several output files in the experiment directory:

- **`model name`** + **`_it_N.mesh`** - Mesh files for each iteration
- **`model name`** + **`_it_N.gf`** - Grid function files for each iteration
- **`parameters.json`** - Configuration and results summary
- **`ParaView/`** - ParaView visualization files (if `--paraview` enabled)

## Performance Considerations

- The system allows multi-threading for parallel error computation
- Number of workers is automatically set to `min(16, cpu_count())`
- Memory usage scales with mesh size and INR complexity. Thus, the nth iteration will take longer than the (n-1)th iteration.
- GPU models require conversion to CPU for processing

## Troubleshooting

### Common Issues

1. **Missing checkpoints**: Ensure model checkpoints exist in the `checkpoints/` directory
2. **Mesh file errors**: Verify source mesh files are in the `meshes/` directory and in the correct format
3. **Memory issues**: Reduce `--max_dofs`, `--error_samples`, or number of workers for large problems
4. **Convergence issues**: Adjust `--threshold` and `--prop_threshold` parameters

## Citation

If you use this code in your research, please cite our preprint 'Pruning AMR: Efficient Visualization of Implicit Neural Representations via Weight Matrix Analysis'

## License


The code in this respository is distributed under the terms of the BSD-3 license.

See [LICENSE](https://github.com/LLNL/PruningAMR/blob/main/LICENSE.md) and
[NOTICE](https://github.com/LLNL/PruningAMR/blob/main/NOTICE.md) for details.

SPDX-License-Identifier: BSD-3

LLNL-CODE-2005503

DOE CODE ID: 165008

## Contribution

Adaptive mesh refinement method using pruning for visualizaiton of implicit nerual representations (INRs)
Version 1.0, September 2025

Code authors: Jennifer Zvonek, Andrew Gillette