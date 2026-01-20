#!/bin/bash

##### These are shell commands
module load cuda/10.2.89
export LD_LIBRARY_PATH=$HOME/.miniconda3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/fast2:$PYTHONPATH
export FVCORE_CACHE=$HOME/storage/torch/iopath_cache

ulimit -n 10000

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
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done


python eval.py \
    --config_path=${CONFIG_PATH} \
    --save_dir=${SAVE_DIR}

echo "Evaluation Completed"
