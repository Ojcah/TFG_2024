#!/bin/bash

python3 -O Data_collection.py --input 'Valores_altos.csv' --output 'Altos_3.csv'

python3 -O Data_collection.py --input 'Altos_1.csv' --output 'Altos_4.csv'

python3 -O Data_collection.py --input 'Altos_3.csv' --output 'Altos_5.csv'

python3 -O Data_collection.py --input 'Valores_altos.csv' --output 'Altos_6.csv'

