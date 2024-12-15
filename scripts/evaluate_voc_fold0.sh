#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/%u/%j.out
#SBATCH --job-name=inference
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --time=2-00:00

nvidia-smi
source activate py38

DATASET=voc
DATA_PATH=/scratch_tmp/users/data/GFSS/VOCdevkit2012/VOC2012/
TRAIN_LIST=./dataset/list/voc/trainaug.txt
VAL_LIST=./dataset/list/voc/val.txt
FOLD=0
SHOT=1
MODEL=pspnet_baseline
BACKBONE=resnet50v2
RESTORE_PATH=/scratch_tmp/users/exp/${DATASET}/${MODEL}/${FOLD}/${BACKBONE}/Novel/${SHOT}/baseline/best_123.pth
BS=1
BASE_SIZE=416,416
OS=8
SAVE=0
SAVE_DIR=./evaluate
SEED=123

python eval_ft.py  --dataset ${DATASET} --data-dir ${DATA_PATH} \
                --train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --test-batch-size ${BS} \
                --model ${MODEL} --restore-from ${RESTORE_PATH} --backbone ${BACKBONE} \
                --base-size ${BASE_SIZE} --save-path ${SAVE_DIR} --save ${SAVE}\
                --fold ${FOLD} --shot ${SHOT} --os ${OS} --random-seed ${SEED}