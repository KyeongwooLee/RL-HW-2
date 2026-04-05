# HW2 Run and Report Guide (Required Experiments Only)

## 1) Environment setup

```bash
conda create -n cas4160 python=3.10
conda activate cas4160
pip install swig==4.0.2
pip install -r requirements.txt
pip install -e .
```

If running headless Linux/Colab for MuJoCo:

```bash
export MUJOCO_GL=egl
```

## 2) Smoke checks (quick local verification)

```bash
python -m cas4160.scripts.run_hw2 --help
python -m cas4160.scripts.run_hw2 --env_name CartPole-v0 --exp_name smoke_pg -n 1 -b 200 -eb 100
python -m cas4160.scripts.run_hw2 --env_name CartPole-v0 --exp_name smoke_baseline -n 1 -b 200 -eb 100 -rtg --use_baseline
python -m cas4160.scripts.run_hw2 --env_name CartPole-v0 --exp_name smoke_gae -n 1 -b 200 -eb 100 -rtg --use_baseline --gae_lambda 0.95
python -m cas4160.scripts.run_hw2 --env_name CartPole-v0 --exp_name smoke_ppo -n 1 -b 200 -eb 100 -rtg --use_baseline --gae_lambda 0.95 --use_ppo --n_ppo_epochs 2 --n_ppo_minibatches 2
```

## 3) Required experiments

### Experiment 1: CartPole (8 runs)

```bash
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
python cas4160/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```

### Experiment 2: HalfCheetah baseline comparison (+1 ablation)

```bash
# no baseline
python cas4160/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --exp_name cheetah

# baseline
python cas4160/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline

# ablation (reduced baseline update)
python cas4160/scripts/run_hw2.py --env_name HalfCheetah-v4 -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 --use_baseline -blr 0.005 -bgs 2 --exp_name cheetah_baseline_lowb
```

### Experiment 3: HumanoidStandup GAE lambda search

```bash
python cas4160/scripts/run_hw2.py --env_name HumanoidStandup-v5 --ep_len 100 --discount 0.99 -n 50 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0 --exp_name HumanoidStandup_lambda0
python cas4160/scripts/run_hw2.py --env_name HumanoidStandup-v5 --ep_len 100 --discount 0.99 -n 50 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name HumanoidStandup_lambda095
python cas4160/scripts/run_hw2.py --env_name HumanoidStandup-v5 --ep_len 100 --discount 0.99 -n 50 -l 3 -s 128 -b 2000 -lr 0.001 --use_reward_to_go --use_baseline --gae_lambda 1 --exp_name HumanoidStandup_lambda1
```

### Experiment 4: Reacher baseline vs PPO

```bash
# baseline
python cas4160/scripts/run_hw2.py --env_name Reacher-v4 --ep_len 1000 --discount 0.99 -n 100 -b 5000 -lr 0.003 -na --use_reward_to_go --use_baseline --gae_lambda 0.97 --exp_name reacher

# PPO
python cas4160/scripts/run_hw2.py --env_name Reacher-v4 --ep_len 1000 --discount 0.99 -n 100 -b 5000 -lr 0.003 -na --use_reward_to_go --use_baseline --gae_lambda 0.97 --use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 --exp_name reacher_ppo
```

## 4) Plot generation rules

- Always use `Train_EnvstepsSoFar` for x-axis.
- Use:
  - `Eval_AverageReturn` for performance plots
  - `Baseline Loss` for baseline training plot

Example:

```bash
python cas4160/scripts/parse_tensorboard.py \
  --input_log_files data/[log1] data/[log2] \
  --human_readable_names "run1" "run2" \
  --data_key "Eval_AverageReturn" \
  --title "Eval Return" \
  --x_label_name "Train Environment Steps" \
  --y_label_name "Eval Return" \
  --output_file "eval_return.png"
```

## 5) Report checklist

- Exp1:
  - 2 plots (small batch, large batch)
  - answers: estimator comparison, advantage normalization impact, batch size impact, exact commands
- Exp2:
  - baseline loss plot, eval return plot
  - effect of reduced `-bgs` and/or `-blr`
- Exp3:
  - single lambda comparison plot
  - explain lambda effect, and meaning of lambda=0 vs lambda=1 with observed performance
- Exp4:
  - baseline vs PPO plot
  - explain why multiple updates without surrogate objective are problematic

## 6) Submission checklist

- Include code + report PDF.
- Exclude videos and raw tensorboard logs.
- Keep zip under 50MB.
