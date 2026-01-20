#!/bin/bash
#SBATCH --account=ncov2019
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --time=24:00:00
#SBATCH --partition=pbatch
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/g/g91/ranganath2/fast2/logs/job-id-%j.txt

##### These are shell commands
module load cuda/10.2.89
export LD_LIBRARY_PATH=$HOME/.miniconda3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/fast2:$PYTHONPATH
export FVCORE_CACHE=$HOME/storage/torch/iopath_cache

export CUDA_VISIBLE_DEVICES=0,1
PREFIX=/g/g91/ranganath2/fast2/logs/
CONFIG_PATH=configs/westnile.yaml
srun -N1 \
    -n1 --exclusive --mpibind=off python eval.py \
    --save_dir=/p/lustre1/ranganath2/fast.tmp/WESTNILE/EGNN-20240819-173525-871557-av89w2z_ \
    --config_path=${CONFIG_PATH} &
srun -N1 \
    -n1 --exclusive --mpibind=off python eval.py \
    --save_dir=/p/lustre1/ranganath2/fast.tmp/DENV2/EGNN-20240819-173536-871558-u0xvdmxn \
    --config_path=${CONFIG_PATH} &
srun -N1 \
    -n1 --exclusive --mpibind=off python eval.py \
    --save_dir=/p/lustre1/ranganath2/fast.tmp/MPRO/EGNN-20240819-173615-871560-_k6nca38 \
    --config_path=${CONFIG_PATH} &
srun -N1 \
    -n1 --exclusive --mpibind=off python eval.py \
    --save_dir=/p/lustre1/ranganath2/fast.tmp/ZIKA/EGNN-20240819-173941-871562-xm0d4ym8 \
    --config_path=${CONFIG_PATH} &
srun -N1 -n1 --exclusive --mpibind=off python eval.py \
    --save_dir=/p/lustre1/ranganath2/fast.tmp/PDBBIND2020/EGNN-20240819-174309-871564-clrte403 \
    --config_path=${CONFIG_PATH}
wait

echo "Evaluation Completed"
