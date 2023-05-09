#!/bin/bash
# python prepare_cityscapes.py \
#     --seed 0 \
#     --data-dir /global/scratch/users/nabeel126/cityscapes/ \
#     --name bbox_square_rand_pad0.2 \
#     --pad 0.2 \
#     --min-area 1000 \
#     --square \
#     --rand-pad \
#     --allow-missing-parts \
#     --use-box-seg

# python prepare_pascal_part.py \
#     --data-dir /global/scratch/users/nabeel126/pascal_part/ \
#     --name aeroplane_bird_car_cat_dog_new \
#     --min-area 0.

# python -u prepare_part_imagenet.py \
#     --data-dir ~/data/PartImageNet/ \
#     --name All 

python -u prepare_paco.py \
    --data-dir ~/data/PACO/ \
    --name All \
    --bbox-expand-factor 1.5 \
    --split_ratios 0.8,0.1,0.1
    
