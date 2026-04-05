#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class ScalarSeries:
    x: List[float]
    y: List[float]
    metric: str
    source: str


def latest_log_dir(data_root: Path, pattern: str) -> Path:
    matches = sorted(data_root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No logs matched pattern: {pattern}")
    return matches[-1]


def resolve_metric_key(acc: EventAccumulator, key: str) -> str:
    tags = set(acc.Tags().get("scalars", []))
    candidates = [key, key.replace(" ", "_"), key.replace("_", " ")]
    for cand in candidates:
        if cand in tags:
            return cand
    raise KeyError(f"Metric '{key}' not found. Available tags: {sorted(tags)}")


def read_scalar_series(log_dir: Path, metric: str) -> ScalarSeries:
    acc = EventAccumulator(str(log_dir))
    acc.Reload()

    env_key = resolve_metric_key(acc, "Train_EnvstepsSoFar")
    metric_key = resolve_metric_key(acc, metric)

    env_events = acc.Scalars(env_key)
    metric_events = acc.Scalars(metric_key)

    env_by_step = {ev.step: float(ev.value) for ev in env_events}

    x = [env_by_step[ev.step] for ev in metric_events]
    y = [float(ev.value) for ev in metric_events]

    return ScalarSeries(x=x, y=y, metric=metric_key, source=log_dir.name)


def summarize(series: ScalarSeries, maximize: bool = True) -> Dict[str, float]:
    ys = np.asarray(series.y, dtype=float)
    xs = np.asarray(series.x, dtype=float)
    idx = int(np.argmax(ys) if maximize else np.argmin(ys))
    return {
        "n_points": int(len(series.y)),
        "first_step": float(xs[0]),
        "last_step": float(xs[-1]),
        "first_value": float(ys[0]),
        "last_value": float(ys[-1]),
        "best_value": float(ys[idx]),
        "best_step": float(xs[idx]),
    }


def plot_lines(
    ax: plt.Axes,
    lines: List[tuple[str, ScalarSeries]],
    title: str,
    y_label: str,
) -> None:
    for label, s in lines:
        ax.plot(s.x, s.y, label=label, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Train Environment Steps")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()


def build_run_index(repo_root: Path) -> Dict[str, Path]:
    data = repo_root / "data"
    return {
        "cartpole": latest_log_dir(data, "q2_pg_cartpole_CartPole-v0_*"),
        "cartpole_rtg": latest_log_dir(data, "q2_pg_cartpole_rtg_CartPole-v0_*"),
        "cartpole_na": latest_log_dir(data, "q2_pg_cartpole_na_CartPole-v0_*"),
        "cartpole_rtg_na": latest_log_dir(data, "q2_pg_cartpole_rtg_na_CartPole-v0_*"),
        "cartpole_lb": latest_log_dir(data, "q2_pg_cartpole_lb_CartPole-v0_*"),
        "cartpole_lb_rtg": latest_log_dir(data, "q2_pg_cartpole_lb_rtg_CartPole-v0_*"),
        "cartpole_lb_na": latest_log_dir(data, "q2_pg_cartpole_lb_na_CartPole-v0_*"),
        "cartpole_lb_rtg_na": latest_log_dir(data, "q2_pg_cartpole_lb_rtg_na_CartPole-v0_*"),
        "cheetah": latest_log_dir(data, "q2_pg_cheetah_HalfCheetah-v4_*"),
        "cheetah_baseline": latest_log_dir(data, "q2_pg_cheetah_baseline_HalfCheetah-v4_*"),
        "cheetah_baseline_lowb": latest_log_dir(data, "q2_pg_cheetah_baseline_lowb_HalfCheetah-v4_*"),
        "humanoid_lambda0": latest_log_dir(data, "q2_pg_HumanoidStandup_lambda0.0_HumanoidStandup-v5_*"),
        "humanoid_lambda095": latest_log_dir(data, "q2_pg_HumanoidStandup_lambda0.95_HumanoidStandup-v5_*"),
        "humanoid_lambda1": latest_log_dir(data, "q2_pg_HumanoidStandup_lambda1.0_HumanoidStandup-v5_*"),
        "reacher": latest_log_dir(data, "q2_pg_reacher_Reacher-v4_*"),
        "reacher_ppo": latest_log_dir(data, "q2_pg_reacher_ppo_Reacher-v4_*"),
    }


def generate_plots(repo_root: Path, output_dir: Path) -> Dict[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = build_run_index(repo_root)
    summary: Dict[str, dict] = {"sources": {k: str(v) for k, v in runs.items()}}

    # Experiment 1
    exp1_small = [
        ("Vanilla", read_scalar_series(runs["cartpole"], "Eval_AverageReturn")),
        ("Reward-to-go", read_scalar_series(runs["cartpole_rtg"], "Eval_AverageReturn")),
        ("Vanilla + AdvNorm", read_scalar_series(runs["cartpole_na"], "Eval_AverageReturn")),
        ("Reward-to-go + AdvNorm", read_scalar_series(runs["cartpole_rtg_na"], "Eval_AverageReturn")),
    ]
    exp1_large = [
        ("Vanilla", read_scalar_series(runs["cartpole_lb"], "Eval_AverageReturn")),
        ("Reward-to-go", read_scalar_series(runs["cartpole_lb_rtg"], "Eval_AverageReturn")),
        ("Vanilla + AdvNorm", read_scalar_series(runs["cartpole_lb_na"], "Eval_AverageReturn")),
        ("Reward-to-go + AdvNorm", read_scalar_series(runs["cartpole_lb_rtg_na"], "Eval_AverageReturn")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_lines(axes[0], exp1_small, "CartPole (batch size = 1000)", "Eval Average Return")
    plot_lines(axes[1], exp1_large, "CartPole (batch size = 4000)", "Eval Average Return")
    fig.tight_layout()
    fig.savefig(output_dir / "exp1_cartpole_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary["exp1_small"] = {name: summarize(s, maximize=True) for name, s in exp1_small}
    summary["exp1_large"] = {name: summarize(s, maximize=True) for name, s in exp1_large}

    # Experiment 2
    exp2_eval = [
        ("No baseline", read_scalar_series(runs["cheetah"], "Eval_AverageReturn")),
        ("Baseline", read_scalar_series(runs["cheetah_baseline"], "Eval_AverageReturn")),
        ("Baseline (reduced updates)", read_scalar_series(runs["cheetah_baseline_lowb"], "Eval_AverageReturn")),
    ]
    exp2_loss = [
        ("Baseline", read_scalar_series(runs["cheetah_baseline"], "Baseline Loss")),
        ("Baseline (reduced updates)", read_scalar_series(runs["cheetah_baseline_lowb"], "Baseline Loss")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_lines(axes[0], exp2_loss, "HalfCheetah Baseline Loss", "Baseline Loss")
    plot_lines(axes[1], exp2_eval, "HalfCheetah Eval Return", "Eval Average Return")
    fig.tight_layout()
    fig.savefig(output_dir / "exp2_cheetah_baseline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary["exp2_eval"] = {name: summarize(s, maximize=True) for name, s in exp2_eval}
    summary["exp2_baseline_loss"] = {
        name: summarize(s, maximize=False) for name, s in exp2_loss
    }

    # Experiment 3
    exp3 = [
        (r"$\lambda=0$", read_scalar_series(runs["humanoid_lambda0"], "Eval_AverageReturn")),
        (r"$\lambda=0.95$", read_scalar_series(runs["humanoid_lambda095"], "Eval_AverageReturn")),
        (r"$\lambda=1$", read_scalar_series(runs["humanoid_lambda1"], "Eval_AverageReturn")),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    plot_lines(ax, exp3, "HumanoidStandup-v5: GAE $\lambda$ Comparison", "Eval Average Return")
    fig.tight_layout()
    fig.savefig(output_dir / "exp3_humanoid_gae.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary["exp3"] = {name: summarize(s, maximize=True) for name, s in exp3}

    # Experiment 4
    exp4 = [
        ("Baseline", read_scalar_series(runs["reacher"], "Eval_AverageReturn")),
        ("PPO", read_scalar_series(runs["reacher_ppo"], "Eval_AverageReturn")),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    plot_lines(ax, exp4, "Reacher-v4: Baseline vs PPO", "Eval Average Return")
    fig.tight_layout()
    fig.savefig(output_dir / "exp4_reacher_ppo.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary["exp4"] = {name: summarize(s, maximize=True) for name, s in exp4}

    summary_path = output_dir / "report_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all HW2 report plots from tensorboard logs.")
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for figures and report_metrics.json (default: <repo_root>/assets).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else repo_root / "assets"

    summary = generate_plots(repo_root, output_dir)
    print(f"Saved plots to: {output_dir}")
    print(f"Saved summary: {output_dir / 'report_metrics.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
