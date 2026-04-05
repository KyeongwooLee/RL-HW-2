#!/bin/bash

set -e

for i in {0.0,0.95,1.0}
do
    python cas4160/scripts/run_hw2.py \
    --env_name HumanoidStandup-v5 --ep_len 100 \
    --discount 0.99 -n 50 -l 3 -s 128 -b 2000 -lr 0.001 \
    --use_reward_to_go --use_baseline --gae_lambda $i \
    --exp_name HumanoidStandup_lambda$i >> Experiments/Ex3/HumanoidStandup_lambda$i.txt
done