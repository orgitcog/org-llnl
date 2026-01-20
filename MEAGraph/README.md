# MEAGraph: Multi-kernel Edge Attention Graph Autoencoder

## Introduction
MEAGraph (Multi-kernel Edge Attention Graph Autoencoder) is a graph-based autoencoder model designed for unsupervised data mining for datasets used in machine learning potentials. It provides accurate clustering for atomic environment identification, unsupervised and unlabeled data pruning for dataset construction.

## Installation

### Requirements
- CPU or NVIDIA GPU
- Linux operating system
- Python 3
- PyTorch and other Python packages described in the instruction below

### Step 1: Python environment (Optional): We recommend using Conda package manager
```bash
conda create -n meagraph python=3.9
source activate meagraph
```

### Step 2: Install PyTorch
```bash
pip install torch==1.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Install PyTorch Geometric and Dependencies
```bash
CUDA=cu116
TORCH=1.13.0
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install dscribe ase networkx matplotlib numpy pandas scikit-learn yacs tqdm pyyaml
```

(Optional) FitSNAP program installation: Refer to [https://fitsnap.github.io/Installation.html](https://fitsnap.github.io/Installation.html). It may be used in case studies for data pruning using bispectrum descriptors. Please see [DataPruning](./run/applications/DataPruning/) in the [applications](./run/applications) folder for details.

### Step 4: Clone and Install MEAGraph
```bash
git clone https://github.com/LLNL/meagraph.git
cd MEAGraph
sh install.sh
```

### Step 5: Test Run

Training (Check [config.py](MEAG_VAE/config.py) for available config settings and their defaults, [configs](./run/configs) folder includes a few examples of user-defined-config.yaml files):
```bash
cd run
python main.py --cfg configs/user-defined-config.yaml
```

Inference: Check the notebooks in [applications](./run/applications) folder to learn how to use `inference.py` for clustering and pruning. See [config.py](MEAG_VAE/config.py) and [args.py](MEAG_VAE/args.py) for available config and argument settings.
```bash
cd run
python inference.py --cfg configs/user-defined-config.yaml --group_name {group_name_strs} --rate {rate} --train_val_ratio {train_val_ratio} --device cpu
```

## Case Studies
The MEAGraph package has been successfully applied to the following case studies (available in the [applications](./run/applications) folder):
- Atomic environment identification for complex dislocation configurations using clustering
- Unsupervised and unlabeled data pruning for Ta dataset and its potential fitting performance
- Understanding how pruning over different cluster sizes affects the force field performance

For more information and detailed case studies, please refer to the documentation and research papers associated with MEAGraph.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) and [NOTICE](NOTICE) file for more details.

SPDX-License-Identifier: MIT

LLNL-CODE-XXXXX

## Citing MEAGraph
If you use MEAGraph in your research or projects, please cite the following paper:

```bibtex
@article{sun2024,
  title={Unsupervised Atomic Data Mining via Multi-Kernel Graph Autoencoders for Machine Learning Potentials},
  author={Sun, Hong and Vita, Joshua A. and Samanta, Amit and Lordi, Vincenzo},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024},
  url={https://arxiv.org/abs/XXXX.XXXXX},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
}
```
