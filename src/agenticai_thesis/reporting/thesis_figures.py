"""Generate publication-ready figures from final experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


class ThesisFigureGenerator:
    """Create the final conventional-versus-Agentic thesis visualizations."""

    MODEL_LABELS = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }
    PIPELINE_LABELS = {"conventional": "Conventional", "agentic": "Agentic AI"}
    COLORS = {"conventional": "#4C78A8", "agentic": "#F58518"}

    def __init__(self, temporal_metrics_path: Path, run_summary_path: Path, output_dir: Path) -> None:
        self._metrics_path = Path(temporal_metrics_path)
        self._summary_path = Path(run_summary_path)
        self._output_dir = Path(output_dir)

    def generate_all(self) -> tuple[Path, ...]:
        """Validate inputs and generate every final figure deterministically."""

        metrics = pd.read_csv(self._metrics_path)
        summary = json.loads(self._summary_path.read_text(encoding="utf-8"))
        self._validate_metrics(metrics)
        self._validate_summary(summary)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        return (
            self._plot_pr_auc(metrics),
            self._plot_precision_recall(metrics),
            self._plot_quality(summary),
            self._plot_runtime(summary),
            self._plot_workflow(),
            self._plot_high_level_architecture(),
        )

    def _plot_pr_auc(self, metrics: pd.DataFrame) -> Path:
        """Compare temporal PR-AUC, the primary imbalanced-data metric."""

        figure, axis = plt.subplots(figsize=(9, 5.5))
        self._grouped_bars(axis, metrics, ("pr_auc",), value_format="{:.3f}")
        axis.set_ylabel("PR-AUC")
        axis.set_ylim(0, max(0.42, metrics["pr_auc"].max() * 1.22))
        axis.set_title("Temporal Holdout PR-AUC: Conventional vs Agentic AI")
        axis.grid(axis="y", alpha=0.2)
        return self._save(figure, "temporal_pr_auc_comparison.png")

    def _plot_precision_recall(self, metrics: pd.DataFrame) -> Path:
        """Show the precision/recall trade-off for each model and pipeline."""

        ordered = self._ordered(metrics)
        models = list(self.MODEL_LABELS)
        x = np.arange(len(models), dtype=float)
        width = 0.18
        figure, axis = plt.subplots(figsize=(10, 5.8))
        offsets = (-1.5, -0.5, 0.5, 1.5)
        series = (
            ("conventional", "precision", "Conventional Precision", "#4C78A8", "//"),
            ("agentic", "precision", "Agentic Precision", "#F58518", "//"),
            ("conventional", "recall", "Conventional Recall", "#72B7B2", ""),
            ("agentic", "recall", "Agentic Recall", "#E45756", ""),
        )
        for offset, (pipeline, metric, label, color, hatch) in zip(offsets, series, strict=True):
            values = [ordered.loc[(model, pipeline), metric] for model in models]
            bars = axis.bar(x + offset * width, values, width, label=label, color=color, hatch=hatch)
            axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2, rotation=90)
        axis.set_xticks(x, [self.MODEL_LABELS[model] for model in models])
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.13)
        axis.set_title("Temporal Holdout Precision and Recall")
        axis.legend(ncol=2, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        return self._save(figure, "temporal_precision_recall_comparison.png")

    def _plot_quality(self, summary: dict[str, object]) -> Path:
        """Visualize the measured data-quality improvement after the Agent plan."""

        values = [summary["initial_data_quality_score"], summary["selected_data_quality_score"]]
        figure, axis = plt.subplots(figsize=(7, 5.5))
        bars = axis.bar(["Before Agent", "After Agent"], values, color=["#9D9D9D", "#59A14F"])
        axis.bar_label(bars, labels=[f"{float(value):.6f}" for value in values], padding=5)
        lower = max(0.0, min(float(value) for value in values) - 0.001)
        axis.set_ylim(lower, 1.0005)
        axis.set_ylabel("Data Quality Score")
        axis.set_title("Data Quality Improvement from the Selected Agentic Plan")
        axis.text(0.5, lower + (1.0005 - lower) * 0.08, f"Δ = {float(summary['selected_quality_delta']):+.6f}", ha="center")
        axis.grid(axis="y", alpha=0.2)
        return self._save(figure, "data_quality_before_after.png")

    def _plot_runtime(self, summary: dict[str, object]) -> Path:
        """Express Agentic computational cost relative to the conventional baseline."""

        values = [1.0, float(summary["selected_runtime_multiplier"])]
        figure, axis = plt.subplots(figsize=(7, 5.5))
        bars = axis.bar(["Conventional", "Agentic AI"], values, color=["#4C78A8", "#F58518"])
        axis.bar_label(bars, labels=[f"{value:.3f}×" for value in values], padding=5)
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        axis.set_ylim(0, max(values) * 1.22)
        axis.set_ylabel("Runtime relative to Conventional")
        axis.set_title("Relative Computational Cost")
        axis.grid(axis="y", alpha=0.2)
        return self._save(figure, "runtime_multiplier_comparison.png")

    def _plot_workflow(self) -> Path:
        """Draw the implemented LangGraph nodes, validation branch and retry loop."""

        figure, axis = plt.subplots(figsize=(13, 7.2))
        axis.set_xlim(0, 13)
        axis.set_ylim(0, 8)
        axis.axis("off")
        nodes = {
            "START": (0.4, 6.1, 1.1, 0.65),
            "Prepare\nIteration": (2.0, 6.0, 1.5, 0.85),
            "Generate\nStrategy": (4.1, 6.0, 1.5, 0.85),
            "Validate\nStrategy": (6.2, 6.0, 1.5, 0.85),
            "Evaluate\nCandidate": (8.5, 6.0, 1.6, 0.85),
            "Assess Invalid\nStrategy": (6.2, 3.7, 1.7, 0.85),
            "Decide\nFeedback": (10.5, 4.8, 1.5, 0.85),
            "Record\nIteration": (8.5, 2.0, 1.6, 0.85),
            "END": (11.3, 2.1, 1.1, 0.65),
        }
        for label, (x, y, width, height) in nodes.items():
            terminal = label in {"START", "END"}
            patch = FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.08",
                facecolor="#E8F1FA" if not terminal else "#D5E8D4",
                edgecolor="#2F4B7C", linewidth=1.5,
            )
            axis.add_patch(patch)
            axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9.5)

        def arrow(start: tuple[float, float], end: tuple[float, float], label: str = "", style: str = "-") -> None:
            axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.5, "linestyle": style, "color": "#444"})
            if label:
                axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.18, label, ha="center", fontsize=8.5)

        arrow((1.5, 6.43), (2.0, 6.43))
        arrow((3.5, 6.43), (4.1, 6.43))
        arrow((5.6, 6.43), (6.2, 6.43))
        arrow((7.7, 6.43), (8.5, 6.43), "valid")
        arrow((6.95, 6.0), (7.0, 4.55), "invalid")
        arrow((10.1, 6.2), (10.8, 5.65))
        arrow((7.9, 4.1), (10.5, 5.0))
        arrow((11.0, 4.8), (9.3, 2.85))
        arrow((10.1, 2.43), (11.3, 2.43), "accept / stop")
        arrow((8.5, 2.35), (2.75, 6.0), "retry", "--")
        axis.set_title("Implemented LangGraph Agentic Data-Quality Workflow", fontsize=15, pad=14)
        axis.text(6.5, 0.65, "Only validated declarative plans reach the deterministic Executor; the temporal holdout opens after plan selection.", ha="center", fontsize=9, color="#555")
        return self._save(figure, "langgraph_agent_workflow.png")

    def _plot_high_level_architecture(self) -> Path:
        """Summarize the complete leakage-safe research workflow in one figure."""

        figure, axis = plt.subplots(figsize=(16, 9))
        axis.set_xlim(0, 16)
        axis.set_ylim(0, 10)
        axis.axis("off")

        # Colors communicate architectural responsibility: data, conventional,
        # Agentic, evaluation and immutable/test-only components.
        colors = {
            "data": "#DCEAF7",
            "conventional": "#D9E8FB",
            "agentic": "#FCE4C5",
            "evaluation": "#DDF0DD",
            "locked": "#E8DDF3",
            "external": "#FFF2B2",
        }

        def box(
            x: float,
            y: float,
            width: float,
            height: float,
            label: str,
            color: str,
            *,
            dashed: bool = False,
            fontsize: float = 10,
        ) -> None:
            patch = FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.08",
                facecolor=color,
                edgecolor="#34495E",
                linewidth=1.6,
                linestyle="--" if dashed else "-",
            )
            axis.add_patch(patch)
            axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize)

        def arrow(
            start: tuple[float, float],
            end: tuple[float, float],
            label: str = "",
            *,
            dashed: bool = False,
        ) -> None:
            axis.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 1.6,
                    "color": "#4A4A4A",
                    "linestyle": "--" if dashed else "-",
                },
            )
            if label:
                axis.text(
                    (start[0] + end[0]) / 2,
                    (start[1] + end[1]) / 2 + 0.18,
                    label,
                    ha="center",
                    fontsize=8.5,
                    color="#444",
                )

        # Data preparation and the leakage boundary.
        box(0.35, 7.7, 1.65, 1.0, "Raw PaySim CSV\n~6.36M rows", colors["data"])
        box(2.55, 7.7, 1.75, 1.0, "Polars Pipeline\nCSV → Parquet", colors["data"])
        box(4.85, 7.7, 2.0, 1.0, "Temporal Split\nby transaction step", colors["data"])
        box(7.55, 8.25, 2.15, 0.9, "Development Data\nmodel/plan selection", colors["data"])
        box(7.55, 6.85, 2.15, 0.9, "Locked Temporal Test\nopened only at the end", colors["locked"], dashed=True)
        arrow((2.0, 8.2), (2.55, 8.2))
        arrow((4.3, 8.2), (4.85, 8.2))
        arrow((6.85, 8.2), (7.55, 8.7))
        arrow((6.85, 8.0), (7.55, 7.3))

        # Two fair experimental branches start from the same development source.
        box(1.0, 4.65, 3.0, 1.25, "CONVENTIONAL BRANCH\nFixed preprocessing pipeline\n(shared 5×5 CV folds)", colors["conventional"], fontsize=10.5)
        box(5.0, 4.65, 3.0, 1.25, "CONTROLLED DEGRADATION\nDeterministic MCAR injection\n(development copy only)", colors["agentic"], fontsize=10.5)
        arrow((8.0, 8.25), (3.3, 5.9))
        arrow((8.7, 8.25), (6.5, 5.9))

        box(5.0, 2.35, 3.0, 1.5, "LANGGRAPH AGENT\nProfile → Generate → Validate\nExecute → Evaluate → Feedback\n(up to 3 iterations)", colors["agentic"], fontsize=10.5)
        box(1.0, 2.55, 3.0, 1.1, "Degraded Conventional\nReference Baseline", colors["conventional"])
        box(8.75, 3.05, 2.25, 1.15, "Groq Free Tier\nGPT-OSS 20B", colors["external"], dashed=True)
        axis.text(9.88, 2.7, "Only compact profiles, metrics\nand JSON plans — never raw rows", ha="center", fontsize=8.2, color="#555")
        arrow((3.0, 4.65), (2.6, 3.65))
        arrow((6.5, 4.65), (6.5, 3.85))
        arrow((8.75, 3.6), (8.0, 3.35), dashed=True)
        arrow((8.0, 2.95), (8.75, 3.35), dashed=True)

        # Identical model family and fold definitions preserve fair comparison.
        box(0.65, 0.55, 3.7, 1.15, "Shared ML Evaluation\nLogistic Regression • Decision Tree • Random Forest\nPR-AUC primary metric", colors["evaluation"], fontsize=9.5)
        box(5.0, 0.55, 2.9, 1.15, "Paired Comparison\nWilcoxon + Holm correction\nquality + predictive + runtime", colors["evaluation"], fontsize=9.5)
        box(8.65, 0.55, 2.6, 1.15, "Selected Agentic Plan\nfrozen before test access", colors["evaluation"])
        box(12.05, 2.9, 3.0, 1.3, "Final Temporal Evaluation\nConventional vs Agentic\non untouched holdout", colors["locked"], fontsize=10.5)
        box(12.05, 0.55, 3.0, 1.15, "Thesis Outputs\nmetrics • statistics • plans\nfigures • audit logs", colors["evaluation"], fontsize=10.5)

        arrow((2.5, 2.55), (2.5, 1.7))
        arrow((5.7, 2.35), (3.9, 1.7))
        arrow((4.35, 1.12), (5.0, 1.12))
        arrow((7.9, 1.12), (8.65, 1.12), "accept")
        arrow((9.7, 6.95), (12.05, 3.75), "test rows")
        arrow((11.25, 1.2), (12.05, 3.1), "frozen plan")
        arrow((13.55, 2.9), (13.55, 1.7))

        axis.set_title("High-Level Architecture of the Complete Experimental Workflow", fontsize=17, pad=16)
        axis.text(
            8,
            9.45,
            "Leakage-safe data preparation, fair Conventional–Agentic comparison and auditable final evaluation",
            ha="center",
            fontsize=10,
            color="#555",
        )
        return self._save(figure, "high_level_system_architecture.png")

    def _grouped_bars(self, axis: plt.Axes, metrics: pd.DataFrame, metric_names: tuple[str, ...], *, value_format: str) -> None:
        ordered = self._ordered(metrics)
        models = list(self.MODEL_LABELS)
        x = np.arange(len(models), dtype=float)
        width = 0.34
        for index, pipeline in enumerate(self.PIPELINE_LABELS):
            values = [ordered.loc[(model, pipeline), metric_names[0]] for model in models]
            bars = axis.bar(x + (index - 0.5) * width, values, width, label=self.PIPELINE_LABELS[pipeline], color=self.COLORS[pipeline])
            axis.bar_label(bars, labels=[value_format.format(value) for value in values], padding=3, fontsize=9)
        axis.set_xticks(x, [self.MODEL_LABELS[model] for model in models])
        axis.legend()

    @staticmethod
    def _ordered(metrics: pd.DataFrame) -> pd.DataFrame:
        return metrics.set_index(["model", "pipeline"])

    @staticmethod
    def _validate_metrics(metrics: pd.DataFrame) -> None:
        required = {"pipeline", "model", "pr_auc", "precision", "recall"}
        missing = sorted(required.difference(metrics.columns))
        if missing:
            raise ValueError(f"Temporal metrics are missing columns: {missing}")
        expected = {(model, pipeline) for model in ThesisFigureGenerator.MODEL_LABELS for pipeline in ThesisFigureGenerator.PIPELINE_LABELS}
        actual = set(zip(metrics["model"], metrics["pipeline"], strict=False))
        if expected != actual:
            raise ValueError("Temporal metrics must contain exactly one row per model and pipeline")

    @staticmethod
    def _validate_summary(summary: dict[str, object]) -> None:
        required = {"initial_data_quality_score", "selected_data_quality_score", "selected_quality_delta", "selected_runtime_multiplier"}
        missing = sorted(required.difference(summary))
        if missing:
            raise ValueError(f"Agentic summary is missing fields: {missing}")

    def _save(self, figure: plt.Figure, filename: str) -> Path:
        path = self._output_dir / filename
        figure.tight_layout()
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return path
