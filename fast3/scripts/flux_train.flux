#!/bin/bash
# Flux scheduler using flux batch
#flux: --output='/p/lustre5/ranganath2/fast2/logs/job-id-{{id}}.txt'
#flux: --error='/p/lustre5/ranganath2/fast2/error-logs/job-id-{{id}}.txt'
#flux: -q=pbatch
#flux: --setattr=thp=always
#flux: --t=1440
#flux: -N=1
#flux: -n=1

export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_XNACK=1

while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -c|--config_path)
      CONFIG_PATH="$2"
      shift # past argument
      ;;
      -d|--save_dir)
      SAVE_DIR="$2"
      shift # past argument
      ;;
      -t|--tag)
      TAG="$2"
      shift # past argument
      ;;
      -r|--resume)
      TUNE="$2"
      shift
      ;;
      -rt|--ray_tune)
      RT="$2"
      shift
      ;;
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done


# Collect arguments


export LD_LIBRARY_PATH=$HOME/.miniconda3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/fast2:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=0,1

ulimit -n 10000

flux run -N 1 -n 1 python train.py \
    --config_path=${CONFIG_PATH} \
    --tag=${TAG} \
    --save_dir=${SAVE_DIR} \
    --tune=${TUNE} \
    --writer="/p/lustre5/ranganath2/fast2/error-logs/job-id-{{id}}.txt"



# Remove all junk
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

echo 'Training Completed!'