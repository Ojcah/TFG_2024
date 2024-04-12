#!/bin/bash

python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [4,4] --model_name gru_mse_simple_b1_h4-4_10 --wandb_run gru_mse_simple_b1_h4-4_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [8,4] --model_name gru_mse_simple_b1_h8-4_10 --wandb_run gru_mse_simple_b1_h8-4_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [32,4] --model_name gru_mse_simple_b1_h32-4_10 --wandb_run gru_mse_simple_b1_h32-4_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [48,4] --model_name gru_mse_simple_b1_h48-4_10 --wandb_run gru_mse_simple_b1_h48-4_10

python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [4,8] --model_name gru_mse_simple_b1_h4-8_10 --wandb_run gru_mse_simple_b1_h4-8_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [8,8] --model_name gru_mse_simple_b1_h8-8_10 --wandb_run gru_mse_simple_b1_h8-8_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [32,8] --model_name gru_mse_simple_b1_h32-8_10 --wandb_run gru_mse_simple_b1_h32-8_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [48,8] --model_name gru_mse_simple_b1_h48-8_10 --wandb_run gru_mse_simple_b1_h48-8_10

python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [4,32] --model_name gru_mse_simple_b1_h4-32_10 --wandb_run gru_mse_simple_b1_h4-32_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [8,32] --model_name gru_mse_simple_b1_h8-32_10 --wandb_run gru_mse_simple_b1_h8-32_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [32,32] --model_name gru_mse_simple_b1_h32-32_10 --wandb_run gru_mse_simple_b1_h32-32_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [48,32] --model_name gru_mse_simple_b1_h48-32_10 --wandb_run gru_mse_simple_b1_h48-32_10

python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [4,48] --model_name gru_mse_simple_b1_h4-48_10 --wandb_run gru_mse_simple_b1_h4-48_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [8,48] --model_name gru_mse_simple_b1_h8-48_10 --wandb_run gru_mse_simple_b1_h8-48_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [32,48] --model_name gru_mse_simple_b1_h32-48_10 --wandb_run gru_mse_simple_b1_h32-48_10
python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size [48,48] --model_name gru_mse_simple_b1_h48-48_10 --wandb_run gru_mse_simple_b1_h48-48_10



# python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size 64 --model_name gru_mse_simple_b1_h64_9 --wandb_run gru_mse_simple_b1_h64_9
# python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size 48 --model_name gru_mse_simple_b1_h48_9 --wandb_run gru_mse_simple_b1_h48_9
# python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size 32 --model_name gru_mse_simple_b1_h32_9 --wandb_run gru_mse_simple_b1_h32_9
# python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size 16 --model_name gru_mse_simple_b1_h16_9 --wandb_run gru_mse_simple_b1_h16_9
# python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size 8 --model_name gru_mse_simple_b1_h8_9 --wandb_run gru_mse_simple_b1_h8_9
# python train_pahm.py --epochs 750 --loss_type mse --batch_size 1 --hidden_size 4 --model_name gru_mse_simple_b1_h4_9 --wandb_run gru_mse_simple_b1_h4_9


#python train_pahm.py --epochs 1500 --loss_type mse --batch_size 1 --hidden_size [32,64,16] --checkpoints_every 500 --model_name gru_mse_simple_b1_32-64-16_8 --extension none --wandb_run gru_mse_simple_b1_32-64-16-8 

#python train_pahm.py --epochs 1500 --loss_type mae --batch_size 1 --checkpoints_every 500 --model_name gru_mae_simple_b1_7 --extension none --wandb_run gru_mae_simple_b1_7 --load_model gru_mse_simple_b1_4_20240407_002119.pth


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

