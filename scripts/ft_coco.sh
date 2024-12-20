#!/bin/bash
uname -a
#date
#env
date

DATASET=coco
DATA_PATH=YOUR_COCO_PATH
TRAIN_LIST=./dataset/list/coco/train.txt
VAL_LIST=./dataset/list/coco/val.txt
FOLD=0
SHOT=1
MODEL=pspnet_ekt
BACKBONE=resnet50v2
RESTORE_PATH=YOUR_RESTORE_PATH
LR=1e-3
WD=1e-4
BS=1
BS_TEST=8
START=0
STEPS=500
BASE_SIZE=473,473
INPUT_SIZE=473,473
OS=8
SEED=123,234
SAVE_DIR=YOUR_SAVE_DIR

cd YOUR_CODE_DIR
python ft_ekt.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --random-seed ${SEED}\
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST}\
			--start-epoch ${START} --num-epoch ${STEPS}\
			--os ${OS} --snapshot-dir ${SAVE_DIR} --save-pred-every 50\
			--fold ${FOLD} --shot ${SHOT} --freeze-backbone --fix-lr --update-base --update-epoch 1
