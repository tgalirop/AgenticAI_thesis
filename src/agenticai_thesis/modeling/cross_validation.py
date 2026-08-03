"""Shared, immutable repeated-stratified cross-validation fold definitions."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


def target_fingerprint(y: np.ndarray) -> str:
    """Create a stable fingerprint of target values and their exact ordering."""

    target = np.ascontiguousarray(np.asarray(y, dtype=np.int64))
    if target.ndim != 1:
        raise ValueError("Target values must be one-dimensional")
    digest = hashlib.sha256()
    digest.update(str(target.shape).encode("ascii"))
    digest.update(target.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FoldSplit:
    """Read-only indices for one repeat/fold combination."""

    repeat: int
    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class CrossValidationFoldSet:
    """Fold collection bound to one exact target vector and configuration."""

    splits: tuple[FoldSplit, ...]
    sample_size: int
    target_sha256: str
    folds: int
    repeats: int
    random_seed: int

    def validate_target(self, y: np.ndarray) -> None:
        """Prevent applying valid fold indices to another sample or row ordering."""

        target = np.asarray(y)
        if target.shape != (self.sample_size,):
            raise ValueError("Target length does not match the shared fold set")
        if target_fingerprint(target) != self.target_sha256:
            raise ValueError("Target values/order do not match the shared fold set fingerprint")


class CrossValidationFoldProvider:
    """Create one reproducible fold set shared by every compared pipeline."""

    def __init__(self, *, folds: int, repeats: int, random_seed: int) -> None:
        if folds < 2:
            raise ValueError("folds must be at least 2")
        if repeats < 1:
            raise ValueError("repeats must be at least 1")
        self._folds = folds
        self._repeats = repeats
        self._random_seed = random_seed

    def create(self, y: np.ndarray) -> CrossValidationFoldSet:
        """Materialise shared splits once and make every index array read-only."""

        target = np.asarray(y, dtype=int)
        if target.ndim != 1:
            raise ValueError("Target values must be one-dimensional")
        classes, counts = np.unique(target, return_counts=True)
        if classes.size != 2:
            raise ValueError("Repeated stratification requires exactly two classes")
        if counts.min() < self._folds:
            raise ValueError("Each class must contain at least one observation per fold")

        splitter = RepeatedStratifiedKFold(
            n_splits=self._folds,
            n_repeats=self._repeats,
            random_state=self._random_seed,
        )
        splits: list[FoldSplit] = []
        # X is unused by the splitter beyond its length. A minimal placeholder
        # avoids duplicating the full feature matrix merely to generate indices.
        placeholder = np.zeros((len(target), 1), dtype=np.uint8)
        for split_index, (train, validation) in enumerate(splitter.split(placeholder, target)):
            train_indices = np.asarray(train, dtype=np.int64)
            validation_indices = np.asarray(validation, dtype=np.int64)
            train_indices.setflags(write=False)
            validation_indices.setflags(write=False)
            splits.append(
                FoldSplit(
                    repeat=split_index // self._folds + 1,
                    fold=split_index % self._folds + 1,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                )
            )
        return CrossValidationFoldSet(
            splits=tuple(splits),
            sample_size=len(target),
            target_sha256=target_fingerprint(target),
            folds=self._folds,
            repeats=self._repeats,
            random_seed=self._random_seed,
        )
