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

### Repository Size Optimization

This repository contains 312,000+ files across 632 projects. To make it accessible:

| Clone Method | Size | Use Case |
|-------------|------|----------|
| Full clone | ~22 GB | Not recommended |
| Shallow + sparse | ~50 MB | Single project work |
| Treeless clone | ~500 MB | Browse multiple projects |
| Blobless clone | ~2 GB | Full history, lazy blobs |

### Recommended: Sparse Checkout (Fastest)

```bash
# Clone with minimal data
git clone --depth 1 --filter=blob:none --sparse https://github.com/orgitcog/org-llnl.git
cd org-llnl

# Checkout only the project you need
git sparse-checkout set 01-hpc-parallel/RAJA

# Add more projects as needed
git sparse-checkout add 06-devtools/Spack
```

### For AI Agents: Treeless Clone

```bash
# Fetch commits but not tree/blob data until needed
git clone --filter=tree:0 https://github.com/orgitcog/org-llnl.git
```

### Browse Without Checkout

```bash
# List all files without downloading content
git ls-tree -r --name-only HEAD

# View file content on demand
git show HEAD:path/to/file.py
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

## Size Contributors (For Maintainers)

The largest directories by file count:

| Directory | Files | Notes |
|-----------|-------|-------|
| `01-hpc-parallel/HPAC` | 94,738 | Full LLVM monorepo fork |
| `06-devtools/SAFIRE` | 28,290 | Large codebase |
| `06-devtools/WVL` | 8,191 | Includes Mac binaries |
| `05-data-viz/MEAGraph` | 8,004 | Run artifacts |
| `test/` directories | 115,567 | Test fixtures across projects |

To further reduce size, consider:
- Converting HPAC to a git submodule referencing upstream LLVM
- Moving large test fixtures to Git LFS or external storage
- Removing build artifacts tracked before .gitignore updates

## Notes

- Each project retains its original license - check individual LICENSE files
- Source: [LLNL GitHub Organization](https://github.com/LLNL)
- See `.gitignore` for excluded file patterns and rationale
