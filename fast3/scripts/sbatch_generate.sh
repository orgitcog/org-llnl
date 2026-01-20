#!/bin/bash
#SBATCH --account=ncov2019
#SBATCH --nodes=1
#SBATCH --partition=pbatch
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/g/g91/ranganath2/fast2/logs/202406052024-generate-hd5.txt

while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -r|--receptor_path)
      RECEPTOR_PATH="$2"
      shift # past argument
      ;;
      -l|--ligand_dir)
      LIGAND_DIR="$2"
      shift # past argument
      ;;
      -o|--output_path)
      OUTPUT_PATH="$2"
      shift # past argument
      ;;
      -c|--csv_path)
      CSV_PATH="$2"
      shift # past argument
      ;;
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done

##### These are shell commands
module load cuda/10.2.89

export CUDA_VISIBLE_DEVICES=0,1

ulimit -n 10000

srun --mpibind=off python -u generate_mlhdf_pdbbind.py \
    --receptor-path=${RECEPTOR_PATH} \
    --ligand-dir=${LIGAND_DIR} \
    --csv-path=${CSV_PATH} \
    --output-path=${OUTPUT_PATH}

#python train.py \
#    --save_dir=./results/PDBBIND2016/CONV3D-20230417-111432-dslqu4mp-NEW_REFINED

# Remove all junk
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

echo 'Training Completed!'