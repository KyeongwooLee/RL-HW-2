#!/bin/bash

set -e

python cas4160/scripts/run_hw2.py \
--env_name Reacher-v4 --ep_len 1000 \
--discount 0.99 -n 100 -b 5000 -lr 0.003 \
-na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
--exp_name reacher >> Experiments/Ex4/reacher.txt
echo "Finished reacher experiment"

python cas4160/scripts/run_hw2.py \
--env_name Reacher-v4 --ep_len 1000 \
--discount 0.99 -n 100 -b 5000 -lr 0.003 \
-na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
--use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 \
--exp_name reacher_ppo >> Experiments/Ex4/reacher_ppo.txt