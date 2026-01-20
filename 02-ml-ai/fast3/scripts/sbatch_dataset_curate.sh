#!/bin/bash
#SBATCH --account=ncov2019
#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/g/g91/ranganath2/fast2/logs/generate-dataset-job-id-%j.txt

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

export PYTHONPATH=$HOME/fast2:$PYTHONPATH


while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -d|--dataset)
      DATASET="$2"
      shift # past argument
      ;;
      -s|--sub-dataset)
      SUBDATASET="$2"
      shift
      ;;
      -o|--output-dir)
      OUTPUTDIR="$2"
      shift
      ;;
      -p|--pocket)
      POCKET="$2"
      shift
      ;;
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done


srun --mpibind=off python -W -u RawToHDF5.py \
      --dataset=${DATASET} \
      --sub-dataset=${SUBDATASET} \
      --output-dir=${OUTPUTDIR} \
      --pocket-type=${POCKET}