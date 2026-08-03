"""Paired statistical comparison of conventional and Agentic CV results."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


class PairedPipelineComparator:
    """Compare pipelines on identical repeat/fold observations."""

    _KEYS = ["model", "repeat", "fold"]

    def compare(
        self,
        conventional: pd.DataFrame,
        agentic: pd.DataFrame,
        *,
        metrics: Sequence[str] = ("pr_auc", "recall", "precision", "f1"),
    ) -> pd.DataFrame:
        """Run two-sided Wilcoxon tests and Holm-correct all p-values."""

        self._validate(conventional, metrics, "conventional")
        self._validate(agentic, metrics, "agentic")
        merged = conventional[[*self._KEYS, *metrics]].merge(
            agentic[[*self._KEYS, *metrics]],
            on=self._KEYS,
            suffixes=("_conventional", "_agentic"),
            validate="one_to_one",
        )
        if len(merged) != len(conventional) or len(merged) != len(agentic):
            raise ValueError("Pipelines do not contain exactly the same model/fold pairs")

        records: list[dict[str, object]] = []
        for model_name, group in merged.groupby("model", sort=False):
            for metric in metrics:
                reference = group[f"{metric}_conventional"].to_numpy(dtype=float)
                candidate = group[f"{metric}_agentic"].to_numpy(dtype=float)
                delta = candidate - reference
                if np.allclose(delta, 0.0):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(candidate, reference, alternative="two-sided")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                records.append(
                    {
                        "model": model_name,
                        "metric": metric,
                        "pairs": len(group),
                        "conventional_mean": float(reference.mean()),
                        "agentic_mean": float(candidate.mean()),
                        "mean_delta": float(delta.mean()),
                        "median_delta": float(np.median(delta)),
                        "agentic_wins": int((delta > 0).sum()),
                        "ties": int(np.isclose(delta, 0.0).sum()),
                        "wilcoxon_statistic": statistic,
                        "p_value": p_value,
                    }
                )
        output = pd.DataFrame.from_records(records)
        output["p_value_holm"] = self._holm(output["p_value"].to_numpy(dtype=float))
        output["significant_0_05"] = output["p_value_holm"] < 0.05
        return output

    @classmethod
    def _validate(cls, frame: pd.DataFrame, metrics: Sequence[str], label: str) -> None:
        required = {*cls._KEYS, *metrics}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} results missing columns: {', '.join(missing)}")
        if frame.duplicated(cls._KEYS).any():
            raise ValueError(f"{label} results contain duplicate model/fold keys")

    @staticmethod
    def _holm(p_values: np.ndarray) -> np.ndarray:
        """Return monotonic Holm-Bonferroni adjusted p-values."""

        order = np.argsort(p_values)
        adjusted = np.empty_like(p_values, dtype=float)
        running_max = 0.0
        total = len(p_values)
        for rank, index in enumerate(order):
            value = min(1.0, (total - rank) * p_values[index])
            running_max = max(running_max, value)
            adjusted[index] = running_max
        return adjusted
