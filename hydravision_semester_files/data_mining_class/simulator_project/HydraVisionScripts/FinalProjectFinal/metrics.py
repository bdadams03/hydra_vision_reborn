#!/usr/bin/env python3
"""
metrics.py
====================
Generates CSV summaries and matplotlib comparison plots from
benchmark results produced by benchmark_runner.py.

Can be used standalone:
    python3 metrics.py --results benchmark_results/results_<ts>.json
"""

import json
import csv
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    import matplotlib
    matplotlib.use("Agg")          # headless-safe backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib/numpy not installed — plots will be skipped.")


# ──────────────────────────────────────────────
# Reporter class
# ──────────────────────────────────────────────

class MetricsReporter:
    """
    Accepts a list of summary dicts (one per model) and an output directory.
    Writes a CSV file and a set of comparison bar-charts.
    """

    METRICS_META = {
        # key: (display_label, unit, higher_is_better)
        "precision":       ("Precision",        "",    True),
        "recall":          ("Recall",            "",    True),
        "f1_score":        ("F1 Score",          "",    True),
        "avg_latency_ms":  ("Avg Latency",       "ms",  False),
        "p95_latency_ms":  ("P95 Latency",       "ms",  False),
        "avg_fps":         ("Avg FPS",           "fps", True),
        "avg_cpu_percent": ("Avg CPU Usage",     "%",   False),
        "avg_ram_mb":      ("Avg RAM Usage",     "MB",  False),
    }

    MODEL_COLORS = {
        "YOLOv8":    "#2196F3",   # blue
        "FasterRCNN": "#FF5722",  # deep orange
        "SSD":        "#4CAF50",  # green
    }

    def __init__(self, summaries: List[Dict], output_dir: str, timestamp: str = ""):
        self.summaries = summaries
        self.output_dir = Path(output_dir)
        self.timestamp = timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ──────────────────────────────────

    def save_csv(self) -> str:
        fname = self.output_dir / f"metrics_{self.timestamp}.csv"
        if not self.summaries:
            print("[Reporter] No summaries to write.")
            return str(fname)

        fieldnames = list(self.summaries[0].keys())
        with open(fname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.summaries)

        print(f"[Reporter] CSV saved → {fname}")
        return str(fname)

    # ── Plots ─────────────────────────────────

    def save_plots(self) -> List[str]:
        if not MATPLOTLIB_AVAILABLE:
            print("[Reporter] matplotlib unavailable — skipping plots.")
            return []

        paths = []
        paths.append(self._plot_grouped_bar())
        paths.append(self._plot_precision_recall_bars())
        paths.append(self._plot_latency_distribution())
        paths.append(self._plot_cpu_ram())
        paths.append(self._plot_radar())
        return [p for p in paths if p]

    # ── grouped bar chart (all metrics) ──────

    def _plot_grouped_bar(self) -> Optional[str]:
        metrics_to_plot = [
            ("precision", "recall", "f1_score"),
            ("avg_latency_ms", "p95_latency_ms"),
            ("avg_fps",),
            ("avg_cpu_percent", "avg_ram_mb"),
        ]
        group_titles = [
            "Detection Quality",
            "Latency (ms)",
            "Throughput (FPS)",
            "Resource Usage",
        ]

        models = [s["model"] for s in self.summaries]
        colors = [self.MODEL_COLORS.get(m, "#888") for m in models]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Object Detection Model Benchmark\n(TurtleBot4 / Gazebo)",
                     fontsize=14, fontweight="bold", y=1.01)

        for ax, keys, title in zip(axes.flat, metrics_to_plot, group_titles):
            x = np.arange(len(models))
            n = len(keys)
            width = 0.6 / n
            for i, key in enumerate(keys):
                vals = [s.get(key, 0) for s in self.summaries]
                bars = ax.bar(x + (i - n / 2 + 0.5) * width, vals,
                              width=width * 0.9,
                              color=[self.MODEL_COLORS.get(m, "#888") for m in models],
                              alpha=0.85, label=self.METRICS_META[key][0])
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005 * max(vals or [1]),
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=9)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        path = str(self.output_dir / f"comparison_grouped_{self.timestamp}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Reporter] Grouped bar chart saved → {path}")
        return path

    # ── precision / recall bars ───────────────

    def _plot_precision_recall_bars(self) -> Optional[str]:
        models = [s["model"] for s in self.summaries]
        precision = [s.get("precision", 0) for s in self.summaries]
        recall    = [s.get("recall", 0)    for s in self.summaries]
        f1        = [s.get("f1_score", 0)  for s in self.summaries]

        x = np.arange(len(models))
        w = 0.25
        fig, ax = plt.subplots(figsize=(9, 5))
        b1 = ax.bar(x - w,   precision, w, label="Precision", color="#1565C0", alpha=0.85)
        b2 = ax.bar(x,       recall,    w, label="Recall",    color="#E65100", alpha=0.85)
        b3 = ax.bar(x + w,   f1,        w, label="F1 Score",  color="#2E7D32", alpha=0.85)

        def _label_bars(bars):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=8)

        for b in (b1, b2, b3):
            _label_bars(b)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1 by Model", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = str(self.output_dir / f"precision_recall_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Reporter] Precision/Recall chart saved → {path}")
        return path

    # ── latency distribution (box-style using raw samples) ──

    def _plot_latency_distribution(self) -> Optional[str]:
        """
        Requires raw inference_times_ms per model.
        Summaries only contain aggregates, so this plot uses avg/p95 if raw
        data isn't available.
        """
        models = [s["model"] for s in self.summaries]
        avgs   = [s.get("avg_latency_ms", 0) for s in self.summaries]
        p95s   = [s.get("p95_latency_ms", 0) for s in self.summaries]

        x = np.arange(len(models))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - w / 2, avgs, w, label="Avg Latency",
                    color=[self.MODEL_COLORS.get(m, "#888") for m in models],
                    alpha=0.85)
        b2 = ax.bar(x + w / 2, p95s, w, label="P95 Latency",
                    color=[self.MODEL_COLORS.get(m, "#888") for m in models],
                    alpha=0.4, edgecolor="black", linewidth=0.8)

        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency (Avg vs P95)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        path = str(self.output_dir / f"latency_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Reporter] Latency chart saved → {path}")
        return path

    # ── CPU & RAM ─────────────────────────────

    def _plot_cpu_ram(self) -> Optional[str]:
        models = [s["model"] for s in self.summaries]
        cpu    = [s.get("avg_cpu_percent", 0) for s in self.summaries]
        ram    = [s.get("avg_ram_mb", 0)      for s in self.summaries]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        colors = [self.MODEL_COLORS.get(m, "#888") for m in models]

        for ax, vals, label, ylabel in (
            (ax1, cpu, "Avg CPU Usage (%)", "%"),
            (ax2, ram, "Avg RAM Usage (MB)", "MB"),
        ):
            bars = ax.bar(models, vals, color=colors, alpha=0.85, width=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}{ylabel}", ha="center", va="bottom", fontsize=9)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle("Resource Utilisation", fontsize=13, fontweight="bold")
        path = str(self.output_dir / f"cpu_ram_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[Reporter] CPU/RAM chart saved → {path}")
        return path

    # ── radar chart ───────────────────────────

    def _plot_radar(self) -> Optional[str]:
        """
        Radar / spider chart normalising all metrics to [0, 1]
        (higher = better, so latency/cpu/ram are inverted).
        """
        radar_keys = ["precision", "recall", "f1_score",
                      "avg_fps", "avg_cpu_percent", "avg_latency_ms"]
        labels = ["Precision", "Recall", "F1", "FPS", "CPU Eff.", "Latency Eff."]
        invert = {k: False for k in radar_keys}
        invert["avg_cpu_percent"] = True
        invert["avg_latency_ms"]  = True

        # Collect raw values
        raw: Dict[str, List[float]] = {k: [s.get(k, 0) for s in self.summaries]
                                        for k in radar_keys}

        # Normalise per metric
        def normalise(vals, inv):
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return [0.5] * len(vals)
            normed = [(v - mn) / (mx - mn) for v in vals]
            if inv:
                normed = [1 - n for n in normed]
            return normed

        normed = {k: normalise(raw[k], invert[k]) for k in radar_keys}

        N = len(radar_keys)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

        for s in self.summaries:
            model = s["model"]
            vals  = [normed[k][self.summaries.index(s)] for k in radar_keys]
            vals += vals[:1]
            color = self.MODEL_COLORS.get(model, "#888")
            ax.plot(angles, vals, color=color, linewidth=2, label=model)
            ax.fill(angles, vals, color=color, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels([])
        ax.set_ylim(0, 1)
        ax.set_title("Model Capability Radar\n(normalised, higher = better)",
                     fontsize=12, fontweight="bold", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(linestyle="--", alpha=0.4)

        path = str(self.output_dir / f"radar_{self.timestamp}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Reporter] Radar chart saved → {path}")
        return path


# ──────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate charts from saved results")
    parser.add_argument("--results", required=True,
                        help="Path to results JSON produced by benchmark_runner.py")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: same as results file)")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    summaries = data["results"]
    output_dir = args.output_dir or str(Path(args.results).parent)
    ts = data.get("timestamp", "standalone")

    reporter = MetricsReporter(summaries, output_dir, ts)
    reporter.save_csv()
    reporter.save_plots()


if __name__ == "__main__":
    main()