#!/bin/bash

set -e

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
--exp_name cartpole >> Experiments/Ex1/cartpole_base.txt

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg --exp_name cartpole_rtg >> Experiments/Ex1/cartpole_rtg.txt

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-na --exp_name cartpole_na >> Experiments/Ex1/cartpole_na.txt

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 \
-rtg -na --exp_name cartpole_rtg_na >> Experiments/Ex1/cartpole_rtg_na.txt

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
--exp_name cartpole_lb >> Experiments/Ex1/cartpole_lb.txt

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
-rtg --exp_name cartpole_lb_rtg >> Experiments/Ex1/cartpole_lb_rtg.txt

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
-na --exp_name cartpole_lb_na >> Experiments/Ex1/cartpole_lb_na.txt

python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 \
-rtg -na --exp_name cartpole_lb_rtg_na >> Experiments/Ex1/cartpole_lb_rtg_na.txt