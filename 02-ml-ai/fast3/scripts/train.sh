#!/bin/bash

##### These are shell commands
module load cuda/10.2.89
export PYTHON_PATH=${PWD}/models/:${PWD}/efficientse3:${PYTHON_PATH}
export FVCORE_CACHE=$HOME/storage/torch/iopath_cache

#export CUDA_VISIBLE_DEVICES=

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
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done

python train.py \
    --config_path=${CONFIG_PATH} \
    --save_dir=${SAVE_DIR} \
    --tag=${TAG} \


# Fusion
#python ../model/fusion_tf/main_fusion_pdbbind.py \
    #--main-dir "pdbbind_fusion" \
    #--fusionmodel-subdir "pdbbind2016_fusion" \
    #--run-mode 3 \
    #--external-csvfile "eval_3dcnn.csv" \
    #--external-3dcnn-featfile "eval_3dcnn_fc10.npy" \
    #--external-sgcnn-featfile "eval_sgcnn_feat.npy" \
    #--external-outprefix "eval_fusion" \
    #--external-dir "pdbbind_2019"

#DATA_DIR=./data/pdbbind2016
DATA_DIR=./data/pdbbind2019_mlhdf

# 3DCNN
#python ./model/3dcnn/main_train.py \
    #--data-dir ${DATA_DIR} \
    #--mlhdf-fn pdbbind2019_refined.hdf \
    #--complex-type 2 \
    #--model-path ./test_3dcnn/model.pth
    ##--csv-fn pdbbind2019_casf_3dcnn.hdf \

# SGCNN
#python train.py \
    #--config_path configs/sgcnn.yaml \
    #--tag=TEST

#python ./model/sgcnn/src/sgcnn/train.py \
    #--checkpoint=true \
    #--checkpoint-dir=test \
    #--num-workers=8 \
    #--batch-size=8 \
    #--preprocessing-type=processed \
    #--feature-type=pybel \
    #--epochs=300 \
    #--lr=1e-3 \
    #--covalent-threshold=1.5 \
    #--non-covalent-threshold=4.5 \
    #--covalent-gather-width=16 \
    #--covalent-k=2 \
    #--non-covalent-gather-width=12 \
    #--non-covalent-k=2 \
    #--checkpoint=True \
    #--checkpoint-iter=100 \
    #--dataset-name pdbbind \
    #--train-data ${DATA_DIR}/general.hdf \
    #--val-data ${DATA_DIR}/refined.hdf


find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf


echo "Training Completed"
