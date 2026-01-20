#!/bin/bash
#SBATCH --account=ncov2019
#SBATCH --nodes=1
#SBATCH --partition=pdebug
#SBATCH --ntasks-per-node=1
#SBATCH --time=11:00:00
#SBATCH --output=/g/g91/ranganath2/fast2/logs/job-id-%j.txt

time=`date +%Y%m%d-%H%M%S`
mv /g/g91/ranganath2/fast2/logs/job-id-${SLURM_JOB_ID}.txt /g/g91/ranganath2/fast2/logs/${SLURM_JOB_ID}-${time}.txt

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

##### These are shell commands
module load rocm


# Collect arguments
source /usr/workspace/${USER}/env/bin/activate

cd /usr/workspace/${USER}/code/fast3/


srun --mpibind=off python -u train.py \
    --config_path=${CONFIG_PATH} \
    --tag=${TAG} \
    --save_dir=${SAVE_DIR} \
    --tune=${TUNE} \
    --writer="/g/g91/ranganath2/fast2/logs/${SLURM_JOB_ID}-${time}.txt"



# Remove all junk
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

echo 'Training Completed!'