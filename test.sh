#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
# eval "$(conda shell.bash hook)"
# conda activate dire

EXP_NAME="girls-new-resnet50"
CKPT="model_epoch_best.pth"
DATASETS_TEST="girls-new"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST arch resnet50 "$@"
