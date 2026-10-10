"""Experiment modules load cleanly and expose the expected research surface."""
import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_looped_vs_deep_arms_have_the_intended_budgets():
    module = load("looped_vs_deep")
    configs = module.arms()
    assert configs["looped-2x"].loops == 2
    assert configs["deep-same-params"].loops == 1
    assert configs["deep-same-params"].stack_blocks == 3
    assert configs["deep-same-applications"].stack_blocks == 6
    from astra_mini import AstraMiniLM

    def block_params(config):
        counts = AstraMiniLM(config).parameter_counts()
        return counts["distinct_block_parameters"], counts["block_applications_per_token"]

    # Param-matched: identical distinct blocks; the loop gains stack_blocks applications.
    looped_blocks, looped_apps = block_params(configs["looped-2x"])
    deep_blocks, deep_apps = block_params(configs["deep-same-params"])
    assert looped_blocks == deep_blocks
    assert looped_apps == deep_apps + configs["looped-2x"].stack_blocks
    # Application-matched: same applications; the deep arm owns the extra distinct blocks.
    deep_apps_blocks, deep_apps_apps = block_params(configs["deep-same-applications"])
    assert deep_apps_apps == looped_apps

    def block_count(config):
        return config.first_blocks + config.stack_blocks + config.last_blocks

    assert block_count(configs["deep-same-applications"]) == block_count(configs["looped-2x"]) + 3
    assert deep_apps_blocks > looped_blocks
    assert looped_blocks == deep_blocks


def test_policy_arms_cover_the_three_policies():
    module = load("loop_depth_policy")
    arms = module.policy_arms()
    assert {config.depth_policy for config in arms.values()} == {"fixed", "exit_gate", "router"}


def test_analyzer_statistics():
    module = load("analyze_loop_experiments")
    assert module.exact_sign_test([0.5, 0.5, 0.5]) < 0.3
    assert module.exact_sign_test([0.1, -0.1, 0.05]) == 1.0
    assert module.exact_sign_test([0.0, 0.0]) == 1.0
    adjusted = module.holm([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]


def test_make_charts_writes_svg(tmp_path):
    spec = importlib.util.spec_from_file_location("make_charts", REPO / "scripts" / "make_charts.py")
    charts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(charts)
    chart = charts.line_chart([1.0, 0.5, 0.2], labels=["0", "10", "20"], y_max=1.1,
                              y_label="loss", title="t", subtitle="s")
    assert chart.startswith("<svg") and chart.rstrip().endswith("</svg>")
