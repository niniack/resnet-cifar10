#!/bin/bash

NAME="sgd_momentum"
ITER=20000

mkdir $NAME

python run.py \
--run_name $NAME \
--checkpoint_dir $NAME \
--lr 0.1 \
--optimizer 'SGD' \
--momentum 0.9 \
--n_iter $ITER \
--save_params_freq 100 \
--decay_lr_1 $(( ITER/2 )) \
--decay_lr_2 $(( 3*ITER/4 )) \
--lr_decay_rate 0.1 \
--wandb 'online'\