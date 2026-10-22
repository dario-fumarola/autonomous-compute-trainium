"""Export aggregate paired learning curves; source text/scores remain local."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compute_trainium.metrics import paired_three_seed_interval


def analyze(experiment):
    progress = json.loads((experiment/"progress.json").read_text())
    records = progress["records"]
    selection = experiment/"selected-lrs.json"
    selected_lrs = json.loads(selection.read_text())["selected_lrs"] if selection.exists() else {}
    directory = experiment/"analysis"
    directory.mkdir(exist_ok=True)
    columns = ["arm", "seed", "peak_lr", "update", "real_tokens", "weighted_nll", "web_nll", "math_nll", "code_nll", "training_matmul_flops"]
    with (directory/"learning-curve.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**{k: record[k] for k in ("arm", "seed", "update", "real_tokens", "training_matmul_flops")},
                             "peak_lr": selected_lrs.get(record["arm"]),
                             "weighted_nll": record["metrics"]["weighted_nll"],
                             **{d+"_nll": record["metrics"]["domains"][d]["nll"] for d in ("web", "math", "code")}})
    paired = []
    for step in sorted({r["update"] for r in records}):
        arms = {arm: {r["seed"]: r for r in records if r["arm"] == arm and r["update"] == step}
                for arm in ("ordinary", "early")}
        common = sorted(arms["ordinary"].keys() & arms["early"].keys())
        if not common:
            continue
        gains = {s: arms["ordinary"][s]["metrics"]["weighted_nll"]-arms["early"][s]["metrics"]["weighted_nll"] for s in common}
        token_counts = {arms[arm][s]["real_tokens"] for arm in arms for s in common}
        if len(token_counts) != 1:
            raise AssertionError("Paired exposures differ")
        row = {"update": step, "real_tokens": token_counts.pop(), "paired_seeds": common,
               "seed_gains_nats": gains, "mean_gain_nats": float(np.mean(list(gains.values())))}
        if len(gains) == 3:
            interval = paired_three_seed_interval(gains)
            row["interval_95"] = interval["interval_95"]
            low, high = row["interval_95"]
            row["assessment_at_this_exposure"] = ("useful_gain_supported" if low > .01 else
                                                   "useful_gain_ruled_out" if high < .01 else "inconclusive")
        else:
            row["assessment_at_this_exposure"] = "preliminary_incomplete_seed_pairs"
        paired.append(row)
    status = "LEARNING_CURVE_COMPLETE" if (experiment/"complete.json").exists() else "LEARNING_CURVE_INTERIM"
    summary = {"status": status, "cohort": "curve", "test_sealed": True,
               "paired": paired, "records": records, "independently_selected_peak_lrs": selected_lrs,
               "scope": "Token-matched conditional on fixed corpus/order; final endpoint is primary; roughly 6.7 tokens/parameter, no convergence or FLOP-efficiency conclusion."}
    (directory/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    # Interim arm means use identical completed seed pairs at each milestone.
    # Keep unpaired completed records in the CSV/JSON, without asymmetric plots.
    plot_records = [r for r in records if any(p["update"] == r["update"] and
                    r["seed"] in p["paired_seeds"] for p in paired)]
    colors = {"ordinary": "#2f5da8", "early": "#d05a30"}
    labels = {"ordinary": "Ordinary", "early": "Pauses from initialization"}
    for arm, lr in selected_lrs.items():
        labels[arm] += f" · peak LR {lr:g}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for arm in ("ordinary", "early"):
        points = []
        for step in sorted({r["update"] for r in plot_records if r["arm"] == arm}):
            pool = [r for r in plot_records if r["arm"] == arm and r["update"] == step]
            points.append((pool[0]["real_tokens"]/1e6, np.mean([r["metrics"]["weighted_nll"] for r in pool])))
        if points:
            x, y = np.array(points).T
            axes[0].plot(x, y, "o-", color=colors[arm], label=labels[arm])
    axes[0].set(xlabel="Real training targets (millions)", ylabel="Unseen-text NLL (nats / real target)")
    if plot_records:
        axes[0].legend(frameon=False, fontsize=8)
    for row in paired:
        error = (np.array(row["interval_95"])-row["mean_gain_nats"])*np.array([-1, 1]) if "interval_95" in row else np.zeros(2)
        axes[1].errorbar(row["real_tokens"]/1e6, row["mean_gain_nats"], yerr=error[:, None],
                         fmt="o", capsize=4, color="#353535")
        axes[1].annotate(f'n={len(row["paired_seeds"])}', (row["real_tokens"]/1e6, row["mean_gain_nats"]),
                         xytext=(5, 5), textcoords="offset points", fontsize=7)
    axes[1].axhline(0, color="#888888", lw=.8)
    axes[1].axhline(.01, color="#888888", ls="--", lw=.8, label="Declared useful gain")
    axes[1].set(xlabel="Real training targets (millions)", ylabel="Ordinary − pause NLL (positive favors pauses)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("150M natural-text pretraining · held-out source groups", fontsize=11)
    fig.savefig(directory/"learning-curve.svg")
    fig.savefig(directory/"learning-curve.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    for axis, domain in zip(axes, ("web", "math", "code")):
        for arm in ("ordinary", "early"):
            pool = sorted((r for r in plot_records if r["arm"] == arm), key=lambda r: r["update"])
            for seed in sorted({r["seed"] for r in pool}):
                rows = [r for r in pool if r["seed"] == seed]
                axis.plot([r["real_tokens"]/1e6 for r in rows],
                          [r["metrics"]["domains"][domain]["nll"] for r in rows],
                          color=colors[arm], alpha=.25, lw=.8)
            points = [(next(r["real_tokens"] for r in pool if r["update"] == step)/1e6,
                       np.mean([r["metrics"]["domains"][domain]["nll"] for r in pool if r["update"] == step]))
                      for step in sorted({r["update"] for r in pool})]
            if points:
                x, y = np.array(points).T
                axis.plot(x, y, "o-", color=colors[arm], label=labels[arm])
        axis.set(title=domain.capitalize(), xlabel="Real targets (millions)", ylabel="NLL (nats / real target)")
        axis.grid(alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
    if plot_records:
        axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out domains · thin lines show initialization seeds", fontsize=11)
    fig.savefig(directory/"domain-curves.svg")
    fig.savefig(directory/"domain-curves.png", dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    analyze(parser.parse_args().experiment)
