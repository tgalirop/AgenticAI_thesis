"""Run the complete conventional preprocessing benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agenticai_thesis.config import PROJECT_ROOT, load_data_config, load_yaml
from agenticai_thesis.modeling.benchmark import (
    benchmark_models,
    load_reproducible_sample,
    summarize_fold_metrics,
)
from agenticai_thesis.modeling.thresholds import save_diagnostic_plots


def main() -> None:
    """Sample development data, run shared CV, and persist every artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    args = parser.parse_args()
    data_config = load_data_config(args.data_config)
    baseline = load_yaml(args.baseline_config)
    seed = int(baseline["cross_validation"]["random_seed"])
    started = time.perf_counter()

    print("1/4 Creating the shared reproducible development sample...")
    sample = load_reproducible_sample(
        data_config.development_path,
        target_column=data_config.target_column,
        negative_to_positive_ratio=int(baseline["sampling"]["negative_to_positive_ratio"]),
        random_seed=seed,
    )
    class_counts = sample[data_config.target_column].value_counts().sort_index().to_dict()
    print(f"Sample rows: {len(sample):,}; class counts: {class_counts}")

    print("2/4 Running identical repeated stratified folds for all models...")
    result = benchmark_models(
        sample,
        model_names=baseline["models"],
        model_parameters=baseline.get("model_parameters", {}),
        target_column=data_config.target_column,
        folds=int(baseline["cross_validation"]["folds"]),
        repeats=int(baseline["cross_validation"]["repeats"]),
        random_seed=seed,
        threshold=float(baseline["decision_threshold"]),
    )

    print("3/4 Writing fold metrics, summaries, and out-of-fold predictions...")
    metrics_directory = PROJECT_ROOT / "reports/metrics"
    metrics_directory.mkdir(parents=True, exist_ok=True)
    result.fold_metrics.to_csv(metrics_directory / "conventional_fold_results.csv", index=False)
    summarize_fold_metrics(result.fold_metrics).to_csv(
        metrics_directory / "conventional_results.csv", index=False
    )
    result.predictions.to_parquet(
        metrics_directory / "conventional_oof_predictions.parquet", index=False
    )

    print("4/4 Creating repeated out-of-fold diagnostic figures...")
    for model_name in baseline["models"]:
        model_predictions = result.predictions[result.predictions["model"] == model_name]
        save_diagnostic_plots(
            model_predictions["y_true"].to_numpy(),
            model_predictions["y_score"].to_numpy(),
            model_name=model_name,
            threshold=float(baseline["decision_threshold"]),
            figures_root=PROJECT_ROOT / "figures",
        )

    log_path = PROJECT_ROOT / "logs/conventional/run_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "sample_rows": len(sample),
                "class_counts": class_counts,
                "folds": int(baseline["cross_validation"]["folds"]),
                "repeats": int(baseline["cross_validation"]["repeats"]),
                "random_seed": seed,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Conventional benchmark completed in {time.perf_counter() - started:.2f} seconds.")


if __name__ == "__main__":
    main()
