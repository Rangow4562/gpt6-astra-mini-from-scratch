#!/usr/bin/env python3
"""Generate SVG charts from receipts (no matplotlib needed)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WIDTH, HEIGHT = 960, 560
PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT = 90, 80, 790, 360
INK, MUTED, GRID, BACKGROUND = "#f4f0e8", "#aaa69d", "#353530", "#171714"
ACCENTS = ("#d97757", "#7aa2c9", "#9eb67b", "#e0b45c")


def svg_document(title: str, subtitle: str, body: list[str]) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
        f'<text x="{PLOT_LEFT}" y="40" fill="{INK}" font-family="Arial" font-size="25" font-weight="700">{title}</text>',
        f'<text x="{PLOT_LEFT}" y="64" fill="{MUTED}" font-family="Arial" font-size="14">{subtitle}</text>',
    ]
    parts.extend(body)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def axes(body: list[str], y_max: float) -> None:
    for tick in range(6):
        value = y_max * tick / 5
        y = PLOT_TOP + PLOT_HEIGHT * (1 - tick / 5)
        body.append(f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_LEFT + PLOT_WIDTH}" y2="{y:.1f}" stroke="{GRID}"/>')
        body.append(f'<text x="{PLOT_LEFT - 12}" y="{y + 5:.1f}" fill="{MUTED}" text-anchor="end" font-family="Arial" font-size="13">{value:g}</text>')


def line_chart(values: list[float], *, labels: list[str], y_max: float, y_label: str,
               title: str, subtitle: str) -> str:
    body = []
    axes(body, y_max)
    label_every = max(1, len(values) // 8)
    points = []
    for index, value in enumerate(values):
        x = PLOT_LEFT + PLOT_WIDTH * index / max(1, len(values) - 1)
        y = PLOT_TOP + PLOT_HEIGHT * (1 - value / y_max)
        points.append(f"{x:.1f},{y:.1f}")
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{ACCENTS[0]}"/>')
        if index % label_every == 0 or index == len(values) - 1:
            label = labels[index] if index < len(labels) else str(index + 1)
            body.append(f'<text x="{x:.1f}" y="{PLOT_TOP + PLOT_HEIGHT + 24}" fill="{MUTED}" text-anchor="middle" font-family="Arial" font-size="13">{label}</text>')
    body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{ACCENTS[0]}" stroke-width="3"/>')
    body.append(f'<text x="22" y="{PLOT_TOP + PLOT_HEIGHT / 2:.0f}" fill="{MUTED}" font-family="Arial" font-size="14" transform="rotate(-90 22 {PLOT_TOP + PLOT_HEIGHT / 2:.0f})">{y_label}</text>')
    return svg_document(title, subtitle, body)


def grouped_bars(groups: list[tuple[str, list[tuple[str, float]]]], *, y_max: float,
                 y_label: str, title: str, subtitle: str) -> str:
    body = []
    axes(body, y_max)
    group_width = PLOT_WIDTH / max(1, len(groups))
    for group_index, (group_label, entries) in enumerate(groups):
        bar_width = min(140, (group_width - 40) / max(1, len(entries)))
        for entry_index, (entry_label, value) in enumerate(entries):
            x = PLOT_LEFT + group_width * group_index + (group_width - bar_width * len(entries)) / 2 + bar_width * entry_index
            y = PLOT_TOP + PLOT_HEIGHT * (1 - value / y_max)
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 8:.1f}" height="{PLOT_TOP + PLOT_HEIGHT - y:.1f}" rx="6" fill="{ACCENTS[entry_index % len(ACCENTS)]}"/>')
            body.append(f'<text x="{x + (bar_width - 8) / 2:.1f}" y="{y - 10:.1f}" fill="{INK}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{value:g}</text>')
            body.append(f'<text x="{x + (bar_width - 8) / 2:.1f}" y="{PLOT_TOP + PLOT_HEIGHT + 24}" fill="{MUTED}" text-anchor="middle" font-family="Arial" font-size="12">{entry_label}</text>')
        body.append(f'<text x="{PLOT_LEFT + group_width * group_index + group_width / 2:.1f}" y="{PLOT_TOP + PLOT_HEIGHT + 46}" fill="{INK}" text-anchor="middle" font-family="Arial" font-size="15" font-weight="700">{group_label}</text>')
    body.append(f'<text x="22" y="{PLOT_TOP + PLOT_HEIGHT / 2:.0f}" fill="{MUTED}" font-family="Arial" font-size="14" transform="rotate(-90 22 {PLOT_TOP + PLOT_HEIGHT / 2:.0f})">{y_label}</text>')
    return svg_document(title, subtitle, body)


def smooth(values: list[float], window: int = 5) -> list[float]:
    return [sum(values[max(0, i - window + 1): i + 1]) / len(values[max(0, i - window + 1): i + 1])
            for i in range(len(values))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/charts"))
    parser.add_argument("--pretrain-receipt", type=Path)
    parser.add_argument("--loop-scaling-receipt", type=Path)
    parser.add_argument("--looped-deep-results", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    written = []

    if args.pretrain_receipt and args.pretrain_receipt.exists():
        receipt = json.loads(args.pretrain_receipt.read_text())
        losses = [row["loss"] for row in receipt["training_curve"]]
        steps = [str(row["step"]) for row in receipt["training_curve"]]
        chart = line_chart(smooth(losses), labels=steps, y_max=max(losses) * 1.1,
                           y_label="Training loss (next byte)",
                           title="Looped pretraining loss curve",
                           subtitle="Next-byte cross-entropy on compositional programs")
        destination = args.output / "pretrain-loss.svg"
        destination.write_text(chart)
        written.append(destination)

    if args.loop_scaling_receipt and args.loop_scaling_receipt.exists():
        receipt = json.loads(args.loop_scaling_receipt.read_text())
        curve = receipt["curve"]
        values = [row["pass_rate"] * 100 for row in curve]
        labels = [f"{row['loops']} loop" + ("s" if row["loops"] > 1 else "") for row in curve]
        chart = line_chart(values, labels=labels, y_max=100, y_label="Held-out pass@1 (%)",
                           title="Test-time loop scaling",
                           subtitle="Same checkpoint, more loops per token at inference")
        destination = args.output / "loop-scaling.svg"
        destination.write_text(chart)
        written.append(destination)

    if args.looped_deep_results and args.looped_deep_results.exists():
        receipt = json.loads(args.looped_deep_results.read_text())
        arms = sorted({row["arm"] for row in receipt["runs"]})
        groups = []
        for arm in arms:
            values = [row["after"]["target_byte_accuracy"] * 100 for row in receipt["runs"] if row["arm"] == arm]
            mean = sum(values) / len(values)
            groups.append((arm, [(f"{mean:.1f}%", mean)]))
        chart = grouped_bars(groups, y_max=100, y_label="Held-out byte accuracy (%)",
                             title="Looped vs deep at matched budgets",
                             subtitle="Param-matched: looped has 2x compute. Application-matched: deep has 2x params.")
        destination = args.output / "looped-vs-deep.svg"
        destination.write_text(chart)
        written.append(destination)

    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
