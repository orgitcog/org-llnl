# LLNL Open Source Gestalt Monorepo

A unified repository containing **632 open-source projects** from Lawrence Livermore National Laboratory (LLNL), organized by domain for efficient agent navigation and integration.

## Quick Navigation for Agents

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

## Agent Instructions

### Finding Projects by Capability

| Need | Domain | Key Projects |
|------|--------|--------------|
| MPI parallelization | `01-hpc-parallel` | `mpi-tools`, `PnMPI`, `mpiBench` |
| GPU computing | `01-hpc-parallel` | `RAJA`, `Umpire`, `CHAI`, `CUDA*` |
| ML/AI models | `02-ml-ai` | `deepopt`, `MLAP`, `Gremlins` |
| Physics simulation | `03-simulation` | `SAMRAI`, `sundials`, `Caliper` |
| Data visualization | `05-data-viz` | `Conduit`, `Ascent`, `VisIt` |
| Build/test tools | `06-devtools` | `Spack`, `BLT`, `RADIUSS` |

### Project Structure Convention

Each project follows this structure:
```
project-name/
├── README.md           # Project documentation
├── LICENSE             # License information
├── CMakeLists.txt      # C/C++ build (if applicable)
├── setup.py            # Python package (if applicable)
├── pyproject.toml      # Modern Python config
├── src/                # Source code
├── include/            # Headers (C/C++)
├── tests/              # Test suite
└── docs/               # Documentation
```

### Language Detection

| File Present | Primary Language |
|--------------|------------------|
| `CMakeLists.txt` | C/C++ |
| `setup.py` or `pyproject.toml` | Python |
| `package.json` | JavaScript/TypeScript |
| `Cargo.toml` | Rust |
| `go.mod` | Go |
| `Makefile` only | C/C++ (legacy) |

## Domain Descriptions

### 01-hpc-parallel (98 projects)
High-Performance Computing frameworks including MPI wrappers, CUDA utilities, GPU programming abstractions, and parallel computing tools. Key projects: RAJA, Umpire, CHAI, Flux.

### 02-ml-ai (127 projects)
Machine learning and artificial intelligence libraries, deep learning frameworks, neural network implementations, and AI-assisted tools.

### 03-simulation (45 projects)
Physics simulation codes, computational fluid dynamics, finite element methods, molecular dynamics, and numerical solvers.

### 04-scientific-libs (11 projects)
Core scientific libraries for mathematics, linear algebra, sparse matrices, and numerical computing foundations.

### 05-data-viz (49 projects)
Data analysis tools, visualization libraries, dashboards, and data management utilities for scientific computing.

### 06-devtools (246 projects)
Developer tools including compilers, debuggers, profilers, testing frameworks, documentation generators, and build systems.

### 07-bioinformatics (5 projects)
Bioinformatics tools for genomics, proteomics, molecular biology, and computational biology applications.

### 08-web-apps (8 projects)
Web applications, frontend frameworks, backend services, and API implementations.

### 09-infrastructure (6 projects)
Infrastructure tools including container definitions, cloud deployment scripts, and DevOps utilities.

### 10-security (2 projects)
Security-focused tools for cryptography, vulnerability analysis, and secure computing.

### 99-misc (35 projects)
Miscellaneous projects that don't fit neatly into other categories.

## Integration Patterns

### Using as Git Submodule
```bash
git submodule add https://github.com/orgitcog/org-llnl.git vendor/llnl
```

### Selective Import (Sparse Checkout)
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/orgitcog/org-llnl.git
cd org-llnl
git sparse-checkout set 01-hpc-parallel/RAJA
```

### CMake Integration
```cmake
add_subdirectory(vendor/llnl/01-hpc-parallel/RAJA)
target_link_libraries(myapp RAJA)
```

### Python Integration
```python
import sys
sys.path.insert(0, 'vendor/llnl/02-ml-ai/project-name')
```

## Index Files

| File | Purpose |
|------|---------|
| `INDEX.json` | Machine-readable project index with metadata |
| `CATALOG.md` | Human-readable project catalog |
| `TAGS.json` | Project tags and keywords for search |

## CI/CD Integration

GitHub Actions workflows included for:
- **Lint Check**: Code style validation
- **Build Matrix**: C/C++ and Python builds
- **Integration Tests**: Cross-project dependency validation
- **Documentation**: Unified docs generation

See `.github/workflows/` for workflow definitions.

## Statistics

| Metric | Value |
|--------|-------|
| Total Projects | 632 |
| C/C++ Projects | 142 |
| Python Projects | 147 |
| JavaScript Projects | 14 |
| Rust Projects | 1 |
| Total Size | ~22 GB |

## Source

Projects sourced from [LLNL GitHub Organization](https://github.com/LLNL).

Individual projects retain their original licenses. See each project's LICENSE file for details.

---

*Generated: January 2026*
*Organization: orgitcog*
