#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=part-model
#SBATCH --gpus=4
#SBATCH --output slurm-part-model-%j.out

ID=2
GPU=0,1,2,3
NUM_GPU=4
BS=64
AA_BS=64
RAND=$(( RANDOM % 10000 ))
PORT="1$(printf '%04d' $RAND)"

# ============================== Part-ImageNet ============================== #
# DATASET="part-imagenet"
# DATAPATH=~/data/PartImageNet
# SEGPATH="$DATAPATH/PartSegmentations/All/"
# ============================== PACO ============================== #
DATASET="paco"
DATAPATH="$HOME/data/PACO"
SEGPATH="$DATAPATH/PartBoxSegmentations/All/"

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901

### Training
EXP_NAME="part-2heads_e-norm_img-semi"
ADV_TRAIN="pgd"
OUTPUT_DIR="./results/$DATASET/$DATASET_NAME/deeplab/$EXP_NAME/$ADV_TRAIN"  # Change as needed

EPOCHS=50

LR=1e-2
WORKERS=8
ATK_STEPS=10

# pretraining
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py --dist-url tcp://localhost:$PORT \
    --seg-backbone "resnet50" --seg-arch "deeplabv3plus" --full-precision --pretrained "imagenet" \
    --seg-label-dir $SEGPATH \
    --dataset $DATASET --batch-size $BS --output-dir $OUTPUT_DIR/pretrained \
    --data $DATAPATH \
    --adv-train none --epochs $EPOCHS --experiment $EXP_NAME \
    --epsilon $EPS \
    --lr $LR \
    --atk-steps $ATK_STEPS \
    --seg-const-trn 0.5 \
    --resume-if-exist \
    --optim "sgd"

# adversarial training
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py --dist-url tcp://localhost:$PORT --workers $WORKERS \
    --seg-backbone "resnet50" --seg-arch "deeplabv3plus" --full-precision --pretrained "imagenet" \
    --dataset $DATASET --batch-size $BS --output-dir $OUTPUT_DIR/advtrained \
    --data $DATAPATH \
    --adv-train $ADV_TRAIN --epochs $EPOCHS --experiment $EXP_NAME \
    --epsilon $EPS \
    --lr $LR \
    --atk-steps $ATK_STEPS \
    --seg-const-trn 0.5 \
    --resume-if-exist \
    --resume $OUTPUT_DIR/pretrained/checkpoint_best.pt --load-weight-only

# evaluation
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py --dist-url tcp://localhost:$PORT \
    --seg-backbone "resnet50" --seg-arch "deeplabv3plus" --full-precision \
    --dataset $DATASET --batch-size $BS --output-dir $OUTPUT_DIR/pretrained \
    --data $DATAPATH \
    --adv-train $ADV_TRAIN --epochs $EPOCHS --experiment $EXP_NAME \
    --epsilon $EPS \
    --lr $LR \
    --seg-const-trn 0.5 \
    --resume $OUTPUT_DIR/pretrained/checkpoint_best.pt --load-weight-only \
    --evaluate
    
CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py --dist-url tcp://localhost:$PORT \
    --seg-backbone "resnet50" --seg-arch "deeplabv3plus" --full-precision \
    --dataset $DATASET --batch-size $BS --output-dir $OUTPUT_DIR/advtrained \
    --data $DATAPATH \
    --adv-train $ADV_TRAIN --epochs $EPOCHS --experiment $EXP_NAME \
    --epsilon $EPS \
    --lr $LR \
    --seg-const-trn 0.5 \
    --optim adamw \
    --resume $OUTPUT_DIR/advtrained/checkpoint_best.pt --load-weight-only \
    --evaluate --eval-attack $ADV_TRAIN