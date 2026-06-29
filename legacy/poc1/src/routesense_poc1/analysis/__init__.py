from .analysis import analyze_records, write_report
from .calibration import evaluate_calibrator, train_calibrator
from .conditional_analysis import (
    analyze_expert_specialization,
    split_by_context_entropy,
    split_by_delta_magnitude,
    split_by_top1_dominance,
)
from .layer_analysis import analyze_by_expert, analyze_by_layer, analyze_oracle_subset, compute_rank_layer_heatmap
from .policies import select_deferrable_expert
from .plotting import plot_deferrable_ratio, plot_policy_delta_nll, plot_ranking_accuracy

__all__ = [
    "analyze_expert_specialization",
    "analyze_by_expert",
    "analyze_by_layer",
    "analyze_oracle_subset",
    "analyze_records",
    "compute_rank_layer_heatmap",
    "evaluate_calibrator",
    "plot_deferrable_ratio",
    "plot_policy_delta_nll",
    "plot_ranking_accuracy",
    "select_deferrable_expert",
    "split_by_context_entropy",
    "split_by_delta_magnitude",
    "split_by_top1_dominance",
    "train_calibrator",
    "write_report",
]
