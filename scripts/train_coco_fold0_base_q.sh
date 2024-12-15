#!/bin/bash -l
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=base
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4G
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=2-00:00
# --constraint=a40

nvidia-smi
source activate py38


DATASET=coco
DATA_PATH=/scratch/data/MSCOCO2014/
TRAIN_LIST=./dataset/list/coco/train.txt
VAL_LIST=./dataset/list/coco/val.txt
FOLD=2
MODEL=pspnet_ekt
BACKBONE=resnet50v2
RESTORE_PATH=/scratch/initmodel/backbones/resnet50_v2.pth
LR=1e-3
WD=1e-4
BS=16
BS_TEST=16
START=0
STEPS=50
BASE_SIZE=473,473
INPUT_SIZE=473,473
OS=8
SEED=123
SAVE_DIR=./exp/${DATASET}/${MODEL}/${FOLD}/${BACKBONE}/base/


#python -m torch.distributed.launch --nproc_per_node=4 
python -m torch.distributed.launch --nproc_per_node=4 train_base.py --dataset ${DATASET} --data-dir ${DATA_PATH} \
			--train-list ${TRAIN_LIST} --val-list ${VAL_LIST} --random-seed ${SEED} \
			--model ${MODEL} --backbone ${BACKBONE} --restore-from ${RESTORE_PATH} \
			--input-size ${INPUT_SIZE} --base-size ${BASE_SIZE} \
			--learning-rate ${LR}  --weight-decay ${WD} --batch-size ${BS} --test-batch-size ${BS_TEST} \
			--start-epoch ${START} --num-epoch ${STEPS} \
			--os ${OS} --snapshot-dir ${SAVE_DIR} --save-pred-every 50 \
			--fold ${FOLD}
