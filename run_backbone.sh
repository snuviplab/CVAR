#!/bin/bash

############### FOR TRAIN ###############
python main.py --mode train \
--dataset_name movielens \
--content "video_only" \
--bsz 512 --epoch 50 --lr 0.0005 \
--warmup_model base \
--save_dir "chkpt/deepfm/movielens/video/lr05" \
--pretrain_model_path "" \
--log_file "train_backbone_video_lr05.log"
