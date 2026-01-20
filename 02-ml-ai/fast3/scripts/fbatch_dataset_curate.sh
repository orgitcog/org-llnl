#!/bin/bash
printf -v date '%(%Y-%m-%d %H:%M:%S)T\n' -1

mv /g/g91/ranganath2/fast2/logs/generate_dataset-job-id-%d.txt /g/g91/ranganath2/fast2/logs/${SLURM_JOB_ID}-${date}.txt



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

filename = "{{name}}-{{jobid}}"

flux run --output= -N 1 -n 1 --exclusive python -u RawToHDF5.py \
      --flux-job-name="curate" \
      --receptor=${RECEPTOR_PATH} \
      --ligand_dir=${LIGAND_DIR} \
      --csv-curate=${CSV} \
      --output-dir=${OUTPUTDIR} \
      --pocket-type=${POCKET}