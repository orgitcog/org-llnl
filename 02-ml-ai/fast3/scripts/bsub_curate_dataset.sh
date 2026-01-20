#!/bin/bash
printf -v date '%(%Y-%m-%d %H:%M:%S)T\n' -1
#SBATCH --account=ncov2019
#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/g/g91/ranganath2/fast2/logs/generate_dataset-${date}.txt

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

export PYTHONPATH=$HOME/fast2:$PYTHONPATH


while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -r|--receptor-path)
      RECEPTOR_PATH="$2"
      shift # past argument
      ;;
      -l|--ligand-dir)
      LIGAND_DIR="$2"
      shift # past argument
      ;;
      -c|--csv-curate)
      CSV="$2"
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


srun --mpibind=off python -u RawToHDF5.py \
    --receptor=${RECEPTOR_PATH} \
    --ligand_dir=${LIGAND_DIR} \
    --csv-curate=${CSV} \
    --output-dir=${OUTPUTDIR} \
    --pocket=${POCKET}