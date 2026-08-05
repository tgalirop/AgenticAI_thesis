"""Run the complete controlled-degradation LangGraph Agentic experiment."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from dotenv import load_dotenv

from agenticai_thesis.agentic.benchmarking import AgenticPlanBenchmark, TemporalHoldoutComparator
from agenticai_thesis.agentic.candidate_evaluation import AgentCandidateEvaluator
from agenticai_thesis.agentic.checkpointing import JsonCheckpointStore
from agenticai_thesis.agentic.domain import ColumnKind, DatasetContext, DatasetRole
from agenticai_thesis.agentic.evaluators import EvaluationConfig, MachineLearningEvaluator
from agenticai_thesis.agentic.execution import ModelPipelineBuilder, TransformationFactoryRegistry
from agenticai_thesis.agentic.executor import TransformationExecutor
from agenticai_thesis.agentic.feedback import (
    CandidateAssessment,
    FeedbackPolicy,
    FeedbackPolicyConfig,
    ModelOutcome,
)
from agenticai_thesis.agentic.graph import AgentGraphDependencies, AgentWorkflow
from agenticai_thesis.agentic.model_clients import GroqClientConfig, GroqModelClient
from agenticai_thesis.agentic.prompts import StrategyPromptProvider
from agenticai_thesis.agentic.state import AgentStateManager, IterationRecord
from agenticai_thesis.agentic.strategy_generator import StrategyGenerator, TransformationPlanParser
from agenticai_thesis.agentic.transformations import TransformationRegistry
from agenticai_thesis.agentic.validator import TransformationPlanValidator
from agenticai_thesis.config import PROJECT_ROOT, load_agent_config, load_data_config, load_yaml
from agenticai_thesis.modeling.benchmark import benchmark_models, load_reproducible_sample, summarize_fold_metrics
from agenticai_thesis.modeling.cross_validation import CrossValidationFoldProvider
from agenticai_thesis.modeling.models import create_estimator
from agenticai_thesis.modeling.preprocessing import CATEGORICAL_FEATURES, MODEL_FEATURES
from agenticai_thesis.modeling.statistical_tests import PairedPipelineComparator
from agenticai_thesis.quality.degradation import ControlledDataDegrader, ControlledDegradationConfig
from agenticai_thesis.quality.plan_quality import TransformationAwarePlanQualityEvaluator
from agenticai_thesis.utils.file_io import write_json_atomic
from agenticai_thesis.utils.resource_monitor import ProcessMemoryMonitor


class AgenticExperimentRunner:
    """Own the reproducible experiment lifecycle from sample to final holdout."""

    def __init__(self, data_config_path: str, baseline_config_path: str, agent_config_path: str) -> None:
        self._data = load_data_config(data_config_path)
        self._baseline = load_yaml(baseline_config_path)
        self._agent_raw = load_yaml(agent_config_path)
        self._agent = load_agent_config(agent_config_path)

    def run(self) -> None:
        started = time.perf_counter()
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        print("1/7 Loading the shared reproducible development sample...")
        sample = load_reproducible_sample(
            self._data.development_path,
            target_column=self._data.target_column,
            negative_to_positive_ratio=int(self._baseline["sampling"]["negative_to_positive_ratio"]),
            random_seed=int(self._baseline["cross_validation"]["random_seed"]),
        )
        print(f"Sample rows: {len(sample):,}")

        print("2/7 Applying controlled degradation to a copy...")
        degradation = self._agent_raw["degradation"]
        degraded_result = ControlledDataDegrader(
            ControlledDegradationConfig(
                missing_rates=degradation["missing_rates"],
                random_seed=int(degradation["random_seed"]),
            ),
            protected_columns=frozenset({self._data.target_column}),
        ).degrade(sample)
        degraded = degraded_result.frame
        self._write_degradation_audit(degraded_result.affected_rows_by_column, len(degraded))

        print("3/7 Evaluating the conventional pipeline on the same degraded folds...")
        with ProcessMemoryMonitor() as baseline_memory_monitor:
            baseline_result = benchmark_models(
                degraded,
                model_names=self._baseline["models"],
                model_parameters=self._baseline.get("model_parameters", {}),
                target_column=self._data.target_column,
                folds=int(self._baseline["cross_validation"]["folds"]),
                repeats=int(self._baseline["cross_validation"]["repeats"]),
                random_seed=int(self._baseline["cross_validation"]["random_seed"]),
                threshold=float(self._baseline["decision_threshold"]),
            )
        baseline_memory = baseline_memory_monitor.summary
        self._save_benchmark("degraded_conventional", baseline_result)

        print("4/7 Running the LangGraph Agent loop...")
        context = self._dataset_context()
        x = degraded[MODEL_FEATURES]
        y = degraded[self._data.target_column].astype(int).to_numpy()
        folds = CrossValidationFoldProvider(
            folds=int(self._baseline["cross_validation"]["folds"]),
            repeats=int(self._baseline["cross_validation"]["repeats"]),
            random_seed=int(self._baseline["cross_validation"]["random_seed"]),
        ).create(y)
        estimators = self._estimators()
        executor = TransformationExecutor(ModelPipelineBuilder(TransformationFactoryRegistry.default()))
        ml_evaluator = MachineLearningEvaluator(
            EvaluationConfig(
                primary_metric=self._baseline["primary_metric"],
                decision_threshold=float(self._baseline["decision_threshold"]),
            )
        )
        quality_evaluator = TransformationAwarePlanQualityEvaluator(x)
        initial_quality = quality_evaluator.initial_quality()
        baseline_assessment = self._baseline_assessment(
            baseline_result.fold_metrics,
            initial_quality.data_quality_score,
            peak_memory_bytes=baseline_memory.peak_rss_bytes,
        )
        registry = TransformationRegistry.default()
        llm = self._agent.llm
        groq_client = GroqModelClient(
            GroqClientConfig(
                model=llm.model,
                base_url=llm.base_url,
                api_key_environment_variable=llm.api_key_environment_variable,
                timeout_seconds=llm.timeout_seconds,
                temperature=llm.temperature,
                reasoning_effort=llm.reasoning_effort,
                max_rate_limit_retries=llm.max_rate_limit_retries,
            )
        )
        strategy_generator = StrategyGenerator(
            model_client=groq_client,
            prompt_provider=StrategyPromptProvider(),
            parser=TransformationPlanParser(),
            allowed_transformations=self._agent.allowed_transformations,
        )
        candidate_evaluator = AgentCandidateEvaluator(
            executor=executor,
            quality_evaluator=quality_evaluator,
            ml_evaluator=ml_evaluator,
            estimators=estimators,
            features=x,
            target=y,
            folds=folds,
        )
        feedback = self._feedback_config()
        state_manager = AgentStateManager()
        run_id = datetime.now(UTC).strftime("agentic_%Y%m%dT%H%M%SZ")
        state = state_manager.create(
            run_id=run_id,
            dataset_id=context.dataset_id,
            baseline=baseline_assessment,
            max_iterations=self._agent.max_iterations,
            initial_data_quality_score=initial_quality.data_quality_score,
        )
        workflow = AgentWorkflow(
            AgentGraphDependencies(
                strategy_generator=strategy_generator,
                validator=TransformationPlanValidator(
                    registry, allowed_transformations=self._agent.allowed_transformations
                ),
                candidate_evaluator=candidate_evaluator,
                feedback_policy=FeedbackPolicy(feedback),
                state_manager=state_manager,
                checkpoint_store=JsonCheckpointStore(PROJECT_ROOT / "artifacts/checkpoints"),
            )
        )
        final_state = workflow.run(
            agent_state=state,
            dataset_context=context,
            quality_metrics={
                "scores": initial_quality.to_dict(),
                "missing_by_column": {
                    column: details["null_count"]
                    for column, details in quality_evaluator.initial_report()["columns"].items()
                    if details["null_count"] > 0
                },
            },
            baseline_metrics=self._baseline_prompt_metrics(baseline_result.fold_metrics),
        )
        self._save_plans(final_state.history)
        selected = self._select_record(final_state.history)
        if not selected.validation.is_valid:
            raise RuntimeError("No valid Agentic plan is available for final evaluation")

        print("5/7 Materializing Agentic fold metrics and statistical tests...")
        agentic_result = AgenticPlanBenchmark(
            executor=executor,
            evaluator=ml_evaluator,
            estimators=estimators,
            features=x,
            target=y,
            folds=folds,
            context=context,
        ).run(selected.validation)
        self._save_benchmark("agentic", agentic_result)
        statistics = PairedPipelineComparator().compare(
            baseline_result.fold_metrics, agentic_result.fold_metrics
        )
        statistics.to_csv(PROJECT_ROOT / "reports/statistical_tests/paired_results.csv", index=False)

        print("6/7 Opening the untouched temporal holdout after plan selection...")
        temporal = pl.read_parquet(self._data.temporal_test_path).to_pandas()
        temporal_output = TemporalHoldoutComparator(
            executor=executor,
            estimators=estimators,
            context=context,
            threshold=float(self._baseline["decision_threshold"]),
        ).evaluate(
            validation=selected.validation,
            development=degraded,
            temporal_test=temporal,
            target_column=self._data.target_column,
        )
        temporal_output.metrics.to_csv(
            PROJECT_ROOT / "reports/metrics/temporal_holdout_results.csv", index=False
        )
        temporal_output.predictions.to_parquet(
            PROJECT_ROOT / "reports/metrics/temporal_holdout_predictions.parquet", index=False
        )

        print("7/7 Writing final run summary...")
        write_json_atomic(
            {
                "run_id": run_id,
                "sample_rows": len(sample),
                "degradation": degraded_result.affected_rows_by_column,
                "initial_data_quality_score": initial_quality.data_quality_score,
                "selected_data_quality_score": selected.assessment.data_quality_score,
                "selected_quality_delta": selected.feedback.quality_delta,
                "selected_primary_metric_delta": selected.feedback.primary_metric_delta,
                "selected_runtime_multiplier": selected.feedback.runtime_multiplier,
                "memory_usage": {
                    "measurement": "sampled process-tree RSS",
                    "degraded_conventional_benchmark": baseline_memory.to_dict(),
                    "selected_agentic_candidate": {
                        "start_rss_bytes": selected.assessment.memory_rss_start_bytes,
                        "peak_rss_bytes": selected.assessment.peak_memory_bytes,
                        "peak_increase_bytes": selected.assessment.memory_peak_increase_bytes,
                        "start_rss_megabytes": (
                            selected.assessment.memory_rss_start_bytes / (1024**2)
                            if selected.assessment.memory_rss_start_bytes is not None
                            else None
                        ),
                        "peak_rss_megabytes": (
                            selected.assessment.peak_memory_bytes / (1024**2)
                            if selected.assessment.peak_memory_bytes is not None
                            else None
                        ),
                        "peak_increase_megabytes": (
                            selected.assessment.memory_peak_increase_bytes / (1024**2)
                            if selected.assessment.memory_peak_increase_bytes is not None
                            else None
                        ),
                    },
                },
                "llm_usage": {
                    **groq_client.usage_summary.to_dict(),
                    "configured_billing_tier": "Groq Free Tier",
                    "observed_api_cost_usd": 0.0,
                    "token_scope": "successful API responses with provider usage metadata",
                },
                "iterations": len(final_state.history),
                "termination_action": final_state.termination_action,
                "selected_plan_id": selected.plan.plan_id,
                "elapsed_seconds": time.perf_counter() - started,
            },
            PROJECT_ROOT / "logs/agentic/run_summary.json",
        )
        print(f"Agentic experiment completed in {time.perf_counter() - started:.2f} seconds.")

    def _dataset_context(self) -> DatasetContext:
        types = {
            feature: (ColumnKind.CATEGORICAL if feature in CATEGORICAL_FEATURES else ColumnKind.NUMERIC)
            for feature in MODEL_FEATURES
        }
        for feature in ("is_transfer", "is_cash_out", "is_merchant_destination"):
            types[feature] = ColumnKind.BINARY
        types[self._data.target_column] = ColumnKind.TARGET
        return DatasetContext(
            dataset_id="paysim-degraded-development-v1",
            role=DatasetRole.DEVELOPMENT,
            column_types=types,
            target_column=self._data.target_column,
            protected_columns=frozenset({self._data.target_column}),
        )

    def _estimators(self) -> dict[str, Any]:
        seed = int(self._baseline["cross_validation"]["random_seed"])
        return {
            name: create_estimator(
                name,
                random_seed=seed,
                parameters=self._baseline.get("model_parameters", {}).get(name, {}),
            )
            for name in self._baseline["models"]
        }

    def _feedback_config(self) -> FeedbackPolicyConfig:
        raw = self._agent_raw["feedback"]
        return FeedbackPolicyConfig(
            max_iterations=self._agent.max_iterations,
            no_improvement_patience=int(self._agent_raw["stopping"]["no_improvement_patience"]),
            **raw,
        )

    @staticmethod
    def _baseline_assessment(
        folds: pd.DataFrame,
        quality_score: float,
        *,
        peak_memory_bytes: int | None = None,
    ) -> CandidateAssessment:
        outcomes = []
        for model, group in folds.groupby("model", sort=False):
            outcomes.append(
                ModelOutcome(
                    model_name=model,
                    status="success",
                    primary_metric=float(group["pr_auc"].mean()),
                    recall=float(group["recall"].mean()),
                    precision=float(group["precision"].mean()),
                    fit_seconds=float(group["fit_seconds"].sum()),
                    predict_seconds=float(group["predict_seconds"].sum()),
                )
            )
        return CandidateAssessment(
            plan_id="degraded_conventional",
            iteration=0,
            model_outcomes=tuple(outcomes),
            data_quality_score=quality_score,
            memory_rss_start_bytes=None,
            peak_memory_bytes=peak_memory_bytes,
            memory_peak_increase_bytes=None,
        )

    @staticmethod
    def _baseline_prompt_metrics(folds: pd.DataFrame) -> dict[str, Any]:
        metrics = {
            model: group[["pr_auc", "recall", "precision"]].mean().to_dict()
            for model, group in folds.groupby("model", sort=False)
        }
        metrics["baseline_pipeline_contract"] = {
            "categorical": "type is one-hot encoded with unknown categories ignored",
            "logistic_regression": "all numeric features are standardized",
            "tree_models": "numeric features pass through without scaling",
            "class_imbalance": "all baseline estimators already use class weights",
        }
        return metrics

    @staticmethod
    def _select_record(history: tuple[IterationRecord, ...]) -> IterationRecord:
        accepted = [record for record in history if record.feedback.action.value == "ACCEPT"]
        if accepted:
            return accepted[-1]
        valid = [
            record
            for record in history
            if record.validation.is_valid and record.assessment.mean_primary_metric is not None
        ]
        if not valid:
            raise RuntimeError("Agent produced no evaluable transformation plan")
        return max(valid, key=lambda record: record.assessment.mean_primary_metric or -1.0)

    @staticmethod
    def _save_benchmark(prefix: str, result: Any) -> None:
        directory = PROJECT_ROOT / "reports/metrics"
        result.fold_metrics.to_csv(directory / f"{prefix}_fold_results.csv", index=False)
        summarize_fold_metrics(result.fold_metrics).to_csv(
            directory / f"{prefix}_results.csv", index=False
        )
        result.predictions.to_parquet(directory / f"{prefix}_oof_predictions.parquet", index=False)

    @staticmethod
    def _save_plans(history: tuple[IterationRecord, ...]) -> None:
        directory = PROJECT_ROOT / "reports/transformation_plans"
        for record in history:
            write_json_atomic(
                record.model_dump(mode="json"), directory / f"{record.plan.plan_id}.json"
            )

    @staticmethod
    def _write_degradation_audit(affected: dict[str, int], rows: int) -> None:
        write_json_atomic(
            {"rows": rows, "affected_rows_by_column": affected},
            PROJECT_ROOT / "reports/profiles/controlled_degradation.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--agent-config", default="configs/agent.yaml")
    args = parser.parse_args()
    AgenticExperimentRunner(args.data_config, args.baseline_config, args.agent_config).run()


if __name__ == "__main__":
    main()
