#!/bin/bash

module load cuda/10.2.89

conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge openbabel
conda install pyg -c pyg
conda install pytorch-sparse -c pyg
conda install numpy
conda install colorlog
conda install h5py
conda install -c dglteam/label/cu102 dgl
conda install conda-forge::e3nn
conda install conda-forge::fvcore
conda install conda-forge::iopath

# cd ..
# conda activate BioML
# python -m pip install -e tfbio
