#!/bin/bash
#SBATCH --job-name=part-model
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --time=48:00:00
#SBATCH --output %j-train-all_fg_fg-sim-maskdino-seg0.0-clf0.0-dtt1-noobj0.1-bg2.out

ID=1
GPU=0,1,2,3
NUM_GPU=4
WORKERS=8
BS=64
AA_BS=64
PORT=1000$ID
BACKEND=nccl

DATASET="paco"
DATASET_NAME="All"
DATAPATH="$HOME/data/PACO"
SEGPATH="$DATAPATH/PartBoxSegmentations/All/"

# OUTPUT_DIR="./results_codebase_changed/$DATASET/$DATASET_NAME/"
OUTPUT_DIR="./results/$DATASET/dino/"  # Change as needed
NUM_SEG_LABELS=-1  # None, 41 (meta partimagenet)

# 0.0156862745, 0.03137254901, 0.06274509803
EPS=0.03137254901
ADV_BETA=1.0
EPOCHS=50
LR=1e-4
WD=5e-4
CSEG=0.0
CCLF=0.0
CDTT=1
NO_OBJECT_WEIGHT=0.1
SEG_ARCH="maskdino"
CONFIG_FILE="configs/maskdino_rn50_part_imagenet.yaml"

EXP="part-sim-semi-learn_mask-1layer"
PRETRAIN_EXP_NAME="$SEG_ARCH-$EXP-lr$LR-wd$WD-seg$CSEG-clf$CCLF-dtt$CDTT-noobj$NO_OBJECT_WEIGHT"

# pretraining
for i in {1..5}; do
    RAND=$(( RANDOM % 10000 ))
    PORT="1$(printf '%04d' $RAND)"
    echo "Trial $i/5: using port $PORT"
    torchrun \
        --rdzv_backend=$BACKEND --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
        --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
        main.py \
        --dist-url tcp://localhost:$PORT --workers $WORKERS \
        --seg-backbone "resnet50" --seg-arch $SEG_ARCH --full-precision \
        --data "$DATAPATH" --seg-label-dir "$SEGPATH" --dataset $DATASET \
        --pretrained --batch-size $BS --epsilon $EPS --atk-norm "Linf" \
        --adv-train "none" --seg-labels $NUM_SEG_LABELS --resume-if-exist \
        --seg-const-trn $CSEG --lr $LR --wd $WD --epochs $EPOCHS \
        --output-dir "$OUTPUT_DIR/pretrained/$PRETRAIN_EXP_NAME" \
        --experiment $EXP \
        --seg-include-bg \
        --config-file $CONFIG_FILE \
        --optim "adamw" --lr-schedule "step" --clip-grad-norm 0.01 \
        --warmup-iters 10 --d2-const-trn $CDTT --clf-const-trn 0.0 \
        MODEL.WEIGHTS "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth" \
        MODEL.MaskDINO.NO_OBJECT_WEIGHT $NO_OBJECT_WEIGHT \
        && break
        sleep 30
done

# adversarial training
train() {
    CCLF=$1
    ADV_EXP_NAME="$SEG_ARCH-$EXP-lr$LR-wd$WD-seg$CSEG-clf$CCLF-dtt$CDTT-noobj$NO_OBJECT_WEIGHT"

    # CUDA_VISIBLE_DEVICES=$GPU python main.py --no-distributed \
    for i in {1..5}; do
        RAND=$(( RANDOM % 10000 ))
        PORT="1$(printf '%04d' $RAND)"
        echo "Trial $i/5: using port $PORT"
        torchrun \
            --rdzv_backend=$BACKEND --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
            --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
            main.py \
            --dist-url tcp://localhost:$PORT --workers $WORKERS \
            --seg-backbone "resnet50" --seg-arch $SEG_ARCH --full-precision \
            --data "$DATAPATH" --seg-label-dir "$SEGPATH" --dataset $DATASET \
            --pretrained --batch-size $BS --epsilon $EPS --atk-norm "Linf" \
            --adv-train "pgd" --adv-beta $ADV_BETA --seg-labels $NUM_SEG_LABELS \
            --seg-const-trn "$CSEG" --lr $LR --wd $WD --epochs $EPOCHS \
            --resume-if-exist \
            --resume $OUTPUT_DIR/pretrained/$PRETRAIN_EXP_NAME/checkpoint_best.pt \
            --output-dir "$OUTPUT_DIR/pgd/$ADV_EXP_NAME" --load-weight-only \
            --experiment $EXP --eval-attack "pgd" \
            --seg-include-bg \
            --config-file $CONFIG_FILE \
            --optim "adamw" --lr-schedule "step" --clip-grad-norm 0.01 \
            --warmup-iters 10 --d2-const-trn $CDTT \
            MODEL.MaskDINO.NO_OBJECT_WEIGHT $NO_OBJECT_WEIGHT \
            && break
        sleep 30
    done
}

train 0.0
# train 0.3
# train 0.7

# evaluate
torchrun \
    --rdzv_backend=$BACKEND --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py \
    --dist-url tcp://localhost:$PORT --workers $WORKERS \
    --seg-backbone "resnet50" --seg-arch $SEG_ARCH --full-precision \
    --data "$DATAPATH" --seg-label-dir "$SEGPATH" --dataset $DATASET \
    --pretrained --batch-size $BS --epsilon $EPS --atk-norm "Linf" \
    --adv-train "none" --seg-labels $NUM_SEG_LABELS --resume-if-exist \
    --seg-const-trn $CSEG --lr $LR --wd $WD --epochs $EPOCHS \
    --output-dir "$OUTPUT_DIR/pretrained/$PRETRAIN_EXP_NAME" \
    --experiment $EXP \
    --seg-include-bg \
    --evaluate \
    --config-file $CONFIG_FILE \
    --optim "adamw" --lr-schedule "step" --clip-grad-norm 0.01 \
    --warmup-iters 10 --d2-const-trn $CDTT --clf-const-trn 0.0 \
    MODEL.WEIGHTS "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth" \
    MODEL.MaskDINO.NO_OBJECT_WEIGHT $NO_OBJECT_WEIGHT 

# CCLF=0.0
ADV_EXP_NAME="$SEG_ARCH-$EXP-lr$LR-wd$WD-seg$CSEG-clf$CCLF-dtt$CDTT-noobj$NO_OBJECT_WEIGHT"
torchrun \
    --rdzv_backend=$BACKEND --rdzv_endpoint=127.0.0.1:2940$ID --rdzv_id=$ID \
    --standalone --nnodes=1 --max_restarts 0 --nproc_per_node=$NUM_GPU \
    main.py \
    --dist-url tcp://localhost:$PORT --workers $WORKERS \
    --seg-backbone "resnet50" --seg-arch $SEG_ARCH --full-precision \
    --data "$DATAPATH" --seg-label-dir "$SEGPATH" --dataset $DATASET \
    --pretrained --batch-size $BS --epsilon $EPS --atk-norm "Linf" \
    --adv-train "pgd" --adv-beta $ADV_BETA --seg-labels $NUM_SEG_LABELS \
    --seg-const-trn "$CSEG" --lr $LR --wd $WD --epochs $EPOCHS \
    --resume-if-exist \
    --resume $OUTPUT_DIR/pretrained/$PRETRAIN_EXP_NAME/checkpoint_best.pt \
    --output-dir "$OUTPUT_DIR/pgd/$ADV_EXP_NAME" --load-weight-only \
    --experiment $EXP --eval-attack "pgd" \
    --seg-include-bg \
    --evaluate --eval-attack "pgd" \
    --config-file $CONFIG_FILE \
    --optim "adamw" --lr-schedule "step" --clip-grad-norm 0.01 \
    --warmup-iters 10 --d2-const-trn $CDTT \
    MODEL.MaskDINO.NO_OBJECT_WEIGHT $NO_OBJECT_WEIGHT \