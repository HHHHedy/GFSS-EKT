#!/bin/bash -l
#SBATCH --output=/scratch_tmp/users/%u/%j.out
#SBATCH --job-name=novel
#SBATCH --gres=gpu
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --time=2-00:00
# --constraint=a40

nvidia-smi
source activate py38


DATASET=voc
DATA_PATH=/scratch_tmp/users/data/GFSS/VOCdevkit2012/VOC2012/
TRAIN_LIST=./dataset/list/voc/trainaug.txt
VAL_LIST=./dataset/list/voc/val.txt
FOLD=0
SHOT=1
MODEL=pspnet_ekt
BACKBONE=resnet50v2
RESTORE_PATH=/scratch_tmp/users/exp/voc/${MODEL}/${FOLD}/${BACKBONE}/base/ours/best.pth 
LR=1e-3
WD=1e-4
BS=1
BS_TEST=16
START=0
STEPS=500
BASE_SIZE=473,473
INPUT_SIZE=473,473
OS=8
SAVE_DIR=/scratch_tmp/users/exp/${DATASET}/${MODEL}/${FOLD}/${BACKBONE}/Novel/${SHOT}/ours
SEED=123


#cd YOUR_CODE_DIR
CUDA_LAUNCH_BLOCKING=1 python ft_ekt.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --random-seed ${SEED}\
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST}\
			--start-epoch ${START} --num-epoch ${STEPS}\
			--os ${OS} --snapshot-dir ${SAVE_DIR} --save-pred-every 50\
			--fold ${FOLD} --shot ${SHOT} --freeze-backbone --fix-lr --update-base --update-epoch 1
