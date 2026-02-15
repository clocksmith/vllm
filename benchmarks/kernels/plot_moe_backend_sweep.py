# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate PNG summaries from MoE backend sweep JSON outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_results(
    paths: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metas: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        meta["source_json"] = path
        metas.append(meta)
        for row in payload.get("results", []):
            row = dict(row)
            row["source_json"] = path
            rows.append(row)
    return metas, rows


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot backend sweep JSON(s) into a summary PNG."
    )
    parser.add_argument(
        "--input-json",
        nargs="+",
        required=True,
        help="One or more JSON outputs from benchmark_moe_backend_sweep.py",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="benchmark_moe_backend_sweep_summary.png",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: "
            "python -m pip install matplotlib"
        ) from exc

    metas, rows = _load_results(args.input_json)
    if not rows:
        raise SystemExit("No results rows found in input JSON(s).")

    # 1) Best strategy counts by phase.
    best_counts: dict[str, Counter[str]] = defaultdict(Counter)
    # 2) Median latency by strategy + phase.
    lat_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    # 3) Sonic speedup by topk + phase.
    speedup_by_topk: dict[tuple[str, int], list[float]] = defaultdict(list)
    # 4) Auto backend counts by phase.
    auto_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for r in rows:
        phase = str(r.get("phase", "unknown"))
        best = str(r.get("best_strategy", "unknown"))
        best_counts[phase][best] += 1

        for strategy, field in (
            ("always_triton", "triton_us"),
            ("always_sonic", "sonic_us"),
            ("pd_split", "pd_split_us"),
        ):
            v = _safe_float(r.get(field))
            if v is not None:
                lat_values[(phase, strategy)].append(v)

        speed = _safe_float(r.get("sonic_speedup_vs_triton"))
        topk = int(r.get("topk", 0))
        if speed is not None:
            speedup_by_topk[(phase, topk)].append(speed)

        auto_backend = str(r.get("auto_backend", "unknown"))
        auto_counts[phase][auto_backend] += 1

    phases = sorted(set(best_counts.keys()) | set(auto_counts.keys()))
    strategies = ["always_triton", "always_sonic", "pd_split"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Plot 1: Best strategy counts by phase.
    x = list(range(len(phases)))
    width = 0.25
    for i, strategy in enumerate(strategies):
        ys = [best_counts[p][strategy] for p in phases]
        ax1.bar([v + (i - 1) * width for v in x], ys, width=width, label=strategy)
    ax1.set_xticks(x, phases)
    ax1.set_title("Best Strategy Count by Phase")
    ax1.set_ylabel("Count")
    ax1.legend()

    # Plot 2: Median latency by strategy and phase.
    x2 = list(range(len(phases)))
    for i, strategy in enumerate(strategies):
        ys = []
        for p in phases:
            m = _median(lat_values[(p, strategy)])
            ys.append(m if m is not None else float("nan"))
        ax2.plot(x2, ys, marker="o", label=strategy)
    ax2.set_xticks(x2, phases)
    ax2.set_title("Median Latency (us) by Phase")
    ax2.set_ylabel("Latency (us)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sonic speedup vs Triton by topk.
    # speedup = triton / sonic ; >1 means Sonic faster.
    topk_values = sorted({k for (_, k) in speedup_by_topk})
    for phase in phases:
        ys = []
        xs = []
        for tk in topk_values:
            m = _median(speedup_by_topk[(phase, tk)])
            if m is None:
                continue
            xs.append(tk)
            ys.append(m)
        if xs:
            ax3.plot(xs, ys, marker="o", label=phase)
    ax3.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax3.set_title("Median Sonic Speedup vs Triton by topk")
    ax3.set_xlabel("topk")
    ax3.set_ylabel("Speedup (triton_us / sonic_us)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Auto backend selection counts by phase.
    backends = sorted({b for c in auto_counts.values() for b in c})
    x4 = list(range(len(phases)))
    width4 = 0.8 / max(1, len(backends))
    for i, backend in enumerate(backends):
        ys = [auto_counts[p][backend] for p in phases]
        ax4.bar(
            [v - 0.4 + i * width4 + width4 / 2 for v in x4],
            ys,
            width=width4,
            label=backend,
        )
    ax4.set_xticks(x4, phases)
    ax4.set_title("Auto Backend Selection Count by Phase")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)

    # Overall title.
    devices = sorted({str(m.get("device", "unknown")) for m in metas})
    title = (
        "MoE Backend Sweep Summary\n"
        f"inputs={len(args.input_json)} rows={len(rows)} devices={', '.join(devices)}"
    )
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output = Path(args.output_png)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Wrote PNG: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
