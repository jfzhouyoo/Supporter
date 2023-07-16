#!/bin/bash

set -e

# set python path according to your actual environment
pythonpath='python'

# forward
${pythonpath} main_rewards.py --direction forward --is_train True --is_test True --learning_rate 2e-5 --gpu 0

# backward
${pythonpath} main_rewards.py --direction backward --is_train True --is_test True --learning_rate 2e-5 --gpu 0
