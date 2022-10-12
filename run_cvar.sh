#!/bin/bash

############### FOR TRAIN ###############
python main.py --mode train --dataset_name movielens \
--content "all" \
--bsz 512 --epoch 1 --lr 0.0005 \
--cvar_epochs 10 --cvar_iters 5 \
--warmup_model cvar \
--save_dir "chkpt/cvar/movielens/all/lr05" \
--pretrain_model_path "pretrained/movielens/all/deepfm-movielens-all.pth" \
--log_file "train_cvar_all_lr05.log"

############### FOR TEST ###############
# python main.py --mode test --dataset_name movielens \
# --content "video_only" --seed 9999 \
# --warmup_model "cvar" \
# --pretrained_base_model_path "pretrained/movielens/video/deepfm-movielens-video.pth" \
# --pretrain_model_path "chkpt/cvar/movielens/video/lr05/best_ndcg_cvar.pth" \
# --log_file "inference_cvar_video_lr05.log"
