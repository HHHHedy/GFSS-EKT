#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/%u/%j.out
#SBATCH --job-name=base
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=2G
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --time=2-00:00
# --constraint=a40

nvidia-smi
source activate py38


DATASET=voc
DATA_PATH=/scratch_tmp/users/data/GFSS/VOCdevkit2012/VOC2012/
TRAIN_LIST=./dataset/list/voc/trainaug.txt
VAL_LIST=./dataset/list/voc/val.txt
FOLD=1
SHOT=1
MODEL=pspnet_ekt
BACKBONE=resnet50v2
RESTORE_PATH=/users/initmodel/backbones/resnet50_v2.pth
LR=1e-3
WD=1e-4
BS=8
BS_TEST=8
START=0
STEPS=50
BASE_SIZE=473,473
INPUT_SIZE=473,473
OS=8
SEED=123
SAVE_DIR=./exp/${DATASET}/${MODEL}/${FOLD}/${BACKBONE}/base/baseline

# cd YOUR_CODE_DIR
python train_base.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --random-seed ${SEED}\
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST}\
			--start-epoch ${START} --num-epoch ${STEPS}\
			--os ${OS} --snapshot-dir ${SAVE_DIR} --save-pred-every 50\
			--fold ${FOLD}