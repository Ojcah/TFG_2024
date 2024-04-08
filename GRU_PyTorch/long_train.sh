#!/bin/bash

python train_pahm.py --epochs 1500 --loss_type mae --batch_size 1 --checkpoints_every 500 --model_name gru_mae_simple_b1_7 --extension none --wandb_run gru_mae_simple_b1_7 --load_model gru_mse_simple_b1_4_20240407_002119.pth


#python train_pahm.py --epochs 1500 --loss_type mae --batch_size 16 --checkpoints_every 500 --model_name gru_mae_simple_b16_5 --extension none --wandb_run gru_mae_simple_b16_5 --load_model gru_mse_simple_b16_20240406_202344.pth
#python train_pahm.py --epochs 1500 --loss_type mae --batch_size 8 --checkpoints_every 500 --model_name gru_mae_simple_b8_5 --extension none --wandb_run gru_mae_simple_b8_5 --load_model gru_mse_simple_b8_4_20240406_233051.pth
#python train_pahm.py --epochs 1500 --loss_type mae --batch_size 1 --checkpoints_every 500 --model_name gru_mae_simple_b1_5 --extension none --wandb_run gru_mae_simple_b1_5 --load_model gru_mse_simple_b1_4_20240407_002119.pth

#python train_pahm.py --epochs 1500 --loss_type mse --batch_size 16 --checkpoints_every 500 --model_name gru_mse_simple_b16_4 --extension none --wandb_run gru_mse_simple_b16_4
#python train_pahm.py --epochs 1500 --loss_type mse --batch_size 8 --checkpoints_every 500 --model_name gru_mse_simple_b8_4 --extension none --wandb_run gru_mse_simple_b8_4
#python train_pahm.py --epochs 1500 --loss_type mse --batch_size 1 --checkpoints_every 500 --model_name gru_mse_simple_b1_4 --extension none --wandb_run gru_mse_simple_b1_4 

#python train_pahm.py --epochs 1500 --loss_type mse --batch_size 16 --checkpoints_every 500 --model_name gru_mse_simple_b16_3 --extension none --wandb_run gru_mse_simple_b16_3 
#python train_pahm.py --epochs 1500 --loss_type mse --checkpoints_every 500 --model_name gru_mse_zero --extension zero --wandb_run gru_mse_zero_1 
#python train_pahm.py --epochs 1500 --loss_type mse --checkpoints_every 500 --model_name gru_mse_one --extension one --wandb_run gru_mse_one_1 
#python train_pahm.py --epochs 1500 --loss_type mse --checkpoints_every 500 --model_name gru_mse_past --extension past --wandb_run gru_mse_past_1

