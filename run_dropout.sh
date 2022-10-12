#!/bin/bash

############### FOR TRAIN ###############
python main.py --mode train --dataset_name movielens \
--content "all" \
--bsz 512 --epoch 30 --lr 0.0005 \
--warmup_model base \
--is_dropoutnet True --dropout_ratio 0.1 \
--save_dir "chkpt/dropoutnet/movielens/all/dr01_lr05" \
--pretrain_model_path "" \
--log_file "train_dr01_all_lr05.log"

############### FOR TEST ###############
# python main.py --mode test --dataset_name movielens \
# --content "all" \
# --pretrain_model_path "chkpt/dropoutnet/movielens/all/dr01_lr05/best_ndcg_dropout.pth" \
# --log_file "inference_dr01_all_lr05.log"
