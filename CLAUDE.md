# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Repository Overview

This is the **LLNL Open Source Gestalt Monorepo** containing 632 open-source projects from Lawrence Livermore National Laboratory, organized into domain-based directories.

## Directory Structure

```
org-llnl/
├── 01-hpc-parallel/     # 98 projects  - MPI, CUDA, GPU, Parallel Computing
├── 02-ml-ai/            # 127 projects - Machine Learning, Deep Learning, AI
├── 03-simulation/       # 45 projects  - Physics, CFD, FEM, Numerical Methods
├── 04-scientific-libs/  # 11 projects  - Math Libraries, Linear Algebra
├── 05-data-viz/         # 49 projects  - Data Analysis, Visualization
├── 06-devtools/         # 246 projects - Compilers, Debuggers, Testing
├── 07-bioinformatics/   # 5 projects   - Genomics, Molecular Biology
├── 08-web-apps/         # 8 projects   - Web Applications, APIs
├── 09-infrastructure/   # 6 projects   - Containers, Cloud, DevOps
├── 10-security/         # 2 projects   - Security, Cryptography
└── 99-misc/             # 35 projects  - Miscellaneous
```

## Key Index Files

- `INDEX.json` - Machine-readable project index with metadata
- `CATALOG.md` - Human-readable project catalog
- `TAGS.json` - Project tags and keywords for search

## Finding Projects

| Need | Domain | Key Projects |
|------|--------|--------------|
| MPI parallelization | `01-hpc-parallel` | `mpi-tools`, `PnMPI`, `mpiBench` |
| GPU computing | `01-hpc-parallel` | `RAJA`, `Umpire`, `CHAI` |
| ML/AI models | `02-ml-ai` | `deepopt`, `MLAP`, `Gremlins` |
| Physics simulation | `03-simulation` | `SAMRAI`, `sundials`, `Caliper` |
| Data visualization | `05-data-viz` | `Conduit`, `Ascent`, `VisIt` |
| Build/test tools | `06-devtools` | `Spack`, `BLT`, `RADIUSS` |

## Language Detection

| File Present | Primary Language |
|--------------|------------------|
| `CMakeLists.txt` | C/C++ |
| `setup.py` or `pyproject.toml` | Python |
| `package.json` | JavaScript/TypeScript |
| `Cargo.toml` | Rust |
| `go.mod` | Go |

## Working with This Repository

Due to the large size (~22 GB), use sparse checkout when working with specific projects:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/orgitcog/org-llnl.git
cd org-llnl
git sparse-checkout set 01-hpc-parallel/RAJA
```

## Project Structure Convention

Individual projects typically follow:
```
project-name/
├── README.md
├── LICENSE
├── CMakeLists.txt or setup.py
├── src/
├── include/
├── tests/
└── docs/
```

## Notes

- Each project retains its original license - check individual LICENSE files
- Source: [LLNL GitHub Organization](https://github.com/LLNL)
