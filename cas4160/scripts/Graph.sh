#!/bin/bash

set -e

echo "Generating graphs 1"
python cas4160/scripts/parse_tensorboard.py \
--input_log_files data/q2_pg_cartpole_CartPole-v0_05-04-2026_07-56-01 data/q2_pg_cartpole_rtg_CartPole-v0_05-04-2026_06-09-19 \
--human_readable_names "Vanilla" "Reward to go" \
--data_key "Eval_AverageReturn" \
--title "Eval Average Return (batch size 1000)" \
--x_label_name "Train Environment Steps" \
--y_label_name "Eval Return" \
--output_file "assets/data/cartpole_plot.png"

echo "Generating graphs 2"
python cas4160/scripts/parse_tensorboard.py \
--input_log_files data/q2_pg_cartpole_na_CartPole-v0_05-04-2026_06-10-04 data/q2_pg_cartpole_rtg_na_CartPole-v0_05-04-2026_06-10-49 \
--human_readable_names "Vanilla" "Reward to go" \
--data_key "Eval_AverageReturn" \
--title "Eval Average Return with Advnorm (batch size 1000)" \
--x_label_name "Train Environment Steps" \
--y_label_name "Eval Return" \
--output_file "assets/data/cartpole_na_plot.png"

echo "Generating graphs 3"
python cas4160/scripts/parse_tensorboard.py \
--input_log_files data/q2_pg_cartpole_lb_CartPole-v0_05-04-2026_06-11-34 data/q2_pg_cartpole_rtg_lb_CartPole-v0_05-04-2026_06-12-19 \
--human_readable_names "Vanilla" "Reward to go" \
--data_key "Eval_AverageReturn" \
--title "Eval Average Return (batch size 4000)" \
--x_label_name "Train Environment Steps" \
--y_label_name "Eval Return" \
--output_file "assets/data/cartpole_lb_plot.png"

echo "Generating graphs 4"
python cas4160/scripts/parse_tensorboard.py \
--input_log_files data/q2_pg_cartpole_lb_na_CartPole-v0_05-04-2026_06-15-36 data/q2_pg_cartpole_rtg_lb_na_CartPole-v0_05-04-2026_06-15-41 \
--human_readable_names "Vanilla" "Reward to go" \
--data_key "Eval_AverageReturn" \
--title "Eval Average Return with Advnorm (batch size 4000)" \
--x_label_name "Train Environment Steps" \
--y_label_name "Eval Return" \
--output_file "assets/data/cartpole_lb_na_plot.png"