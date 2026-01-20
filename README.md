# LLNL Open Source Gestalt Monorepo

This repository integrates **633 open-source projects** from Lawrence Livermore National Laboratory (LLNL) into a cohesive monorepo structure for unified access, analysis, and development.

## Overview

Lawrence Livermore National Laboratory is a premier research institution focused on national security, energy, and scientific discovery. This monorepo consolidates their extensive open-source portfolio spanning:

- **High-Performance Computing (HPC)** frameworks and tools
- **Scientific Simulation** codes and solvers
- **Machine Learning & AI** libraries
- **Performance Analysis** and profiling tools
- **Numerical Methods** and mathematical libraries
- **Data Management** and visualization tools
- **System Administration** utilities

## Repository Structure

Each subdirectory corresponds to an individual LLNL project, preserving its original structure:

```
org-llnl/
├── RAJA/                 # Performance Portability Layer (C++)
├── sundials/             # Nonlinear/Differential Equation Solvers
├── Caliper/              # Performance Profiling Library
├── conduit/              # Simplified Data Exchange for HPC
├── axom/                 # CS Infrastructure for HPC Applications
├── zfp/                  # Compressed Numerical Arrays
├── Umpire/               # Memory Management for NUMA & GPU
├── merlin/               # ML for HPC Workflows
├── hypre/                # Parallel Sparse Linear Solvers
├── ... (633 projects total)
└── README.md
```

## Key Project Categories

### Performance Portability
- **RAJA** - Performance portability layer for C++
- **CHAI** - Copy-hiding array abstraction for memory spaces
- **Umpire** - Memory management for heterogeneous architectures
- **camp** - Compiler agnostic metaprogramming library

### Numerical Solvers
- **sundials** - Suite of nonlinear and differential/algebraic solvers
- **hypre** - Parallel multigrid solvers
- **hiop** - HPC nonlinear optimization solver
- **libROM** - Data-driven model reduction library

### Simulation Codes
- **WarpX** - Electromagnetic Particle-In-Cell code
- **SAMRAI** - Structured Adaptive Mesh Refinement
- **spheral** - Meshfree hydrodynamics
- **ExaCA** - Cellular automata for alloy solidification

### Performance Analysis
- **Caliper** - Instrumentation and profiling library
- **hatchet** - Graph-indexed performance data analysis
- **STAT** - Stack Trace Analysis Tool
- **mpiP** - Lightweight MPI profiler

### Machine Learning & AI
- **merlin** - ML for HPC workflows
- **GPLaSDI** - Gaussian process latent space dynamics
- **MuyGPyS** - Fast Gaussian process implementation
- **DJINN** - Deep jointly-informed neural networks

### Data Management
- **conduit** - Simplified data exchange for HPC
- **Silo** - Mesh and field I/O library
- **kosh** - Data storage and query via Python
- **Sina** - Simulation metadata management

## Integration Notes

This monorepo was created by:
1. Cloning all 633 LLNL repositories with shallow depth
2. Removing individual `.git` directories
3. Integrating into a unified version control structure

Original repository histories are preserved in the source LLNL GitHub organization: https://github.com/LLNL

## License

Individual projects retain their original licenses. Most LLNL projects use permissive open-source licenses (MIT, BSD, Apache 2.0). Refer to each project's LICENSE file for specifics.

## Contributing

For contributions to individual projects, please refer to the original LLNL repositories. This monorepo serves as a consolidated reference and integration point.

## Links

- **LLNL GitHub**: https://github.com/LLNL
- **LLNL Software Portal**: https://software.llnl.gov
- **RADIUSS Project**: https://computing.llnl.gov/projects/radiuss

---

*Generated: January 2026*
*Total Projects: 633*
*Source: Lawrence Livermore National Laboratory Open Source Portfolio*
