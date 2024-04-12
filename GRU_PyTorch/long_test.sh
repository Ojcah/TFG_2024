#!/bin/bash

python train_pahm.py --model_name gru_mse_simple_b1_h4-4_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h8-4_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h32-4_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h48-4_10 --no_fig

python train_pahm.py --model_name gru_mse_simple_b1_h4-8_10 --no_fig 
python train_pahm.py --model_name gru_mse_simple_b1_h8-8_10 --no_fig 
python train_pahm.py --model_name gru_mse_simple_b1_h32-8_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h48-8_10 --no_fig

python train_pahm.py --model_name gru_mse_simple_b1_h4-32_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h8-32_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h32-32_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h48-32_10 --no_fig

python train_pahm.py --model_name gru_mse_simple_b1_h4-48_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h8-48_10 --no_fig 
python train_pahm.py --model_name gru_mse_simple_b1_h32-48_10 --no_fig
python train_pahm.py --model_name gru_mse_simple_b1_h48-48_10 --no_fig
