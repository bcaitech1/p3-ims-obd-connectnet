#!/bin/sh
echo "#########################################################################################" 
python ./train.py --data_ratio 4 
echo "#########################################################################################" 
python ./train.py --data_ratio 3 
echo "#########################################################################################" 
python ./train.py --data_ratio 2 
echo "#########################################################################################" 
python ./train.py --data_ratio 1 
echo "#########################################################################################" 
python ./train.py --data_ratio 0  
