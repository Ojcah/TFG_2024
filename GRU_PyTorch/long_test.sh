#!/bin/bash

for i in gru_mse_simple_b1_4_20240407_002119.pth \
         gru_mse_simple_b16_20240406_202344.pth \
         gru_mse_simple_b16_2_20240406_204708.pth \
         gru_mse_simple_b16_3_20240406_224638.pth \
         gru_mse_simple_b16_4_20240406_231352.pth \
         gru_mse_simple_b1_h16_9_20240410_192314.pth \
         gru_mse_simple_b1_h32-32_10_20240411_123209.pth \
         gru_mse_simple_b1_h32-4_10_20240411_083731.pth \
         gru_mse_simple_b1_h32-48_10_20240411_142922.pth \
         gru_mse_simple_b1_h32-8_10_20240411_103424.pth \
         gru_mse_simple_b1_h32_9_20240410_185234.pth \
         gru_mse_simple_b1_h4-32_10_20240411_113259.pth \
         gru_mse_simple_b1_h4-4_10_20240411_073843.pth \
         gru_mse_simple_b1_h4-48_10_20240411_133148.pth \
         gru_mse_simple_b1_h4-8_10_20240411_093457.pth \
         gru_mse_simple_b1_h48-32_10_20240411_130152.pth \
         gru_mse_simple_b1_h48-4_10_20240411_090612.pth \
         gru_mse_simple_b1_h48-48_10_20240411_145821.pth \
         gru_mse_simple_b1_h48-8_10_20240411_110345.pth \
         gru_mse_simple_b1_h48_9_20240410_182133.pth \
         gru_mse_simple_b1_h4_9_20240410_202500.pth \
         gru_mse_simple_b1_h64_9_20240410_175108.pth \
         gru_mse_simple_b1_h8-32_10_20240411_120233.pth \
         gru_mse_simple_b1_h8-4_10_20240411_080812.pth \
         gru_mse_simple_b1_h8-48_10_20240411_140050.pth \
         gru_mse_simple_b1_h8-8_10_20240411_100520.pth \
         gru_mse_simple_b1_h8_9_20240410_195454.pth
do
    python test_pahm.py --model_name $i --no_fig
    echo
done
