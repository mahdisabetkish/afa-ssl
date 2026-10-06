"""Seeded stratified test split, nested stratified labeled subsets, standardisation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stratified_order(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation whose every prefix is approximately stratified. Each row gets
    u = (rank within its shuffled class + 1) / class size; sort by u."""
    u = np.empty(len(y))
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        u[idx] = (np.arange(len(idx)) + 1 + rng.random(len(idx)) * 1e-3) / len(idx)
    return np.argsort(u)


@dataclass
class Split:
    test: np.ndarray
    pool: np.ndarray          # training pool (features visible, labels hidden)
    pool_order: np.ndarray    # stratified nested ordering of the pool
    mean: np.ndarray
    std: np.ndarray

    def labeled(self, n: int) -> np.ndarray:
        n = min(n, len(self.pool_order))
        return self.pool_order[:n]


def make_split(y: np.ndarray, n_test: int, rng: np.random.Generator, X: np.ndarray,
               pool_mask: np.ndarray | None = None, test_mask: np.ndarray | None = None) -> Split:
    """Draw `n_test` stratified test rows, the rest is the pool. Optional cohort
    masks restrict which rows may go to the pool and to the test set."""
    n = len(y)
    has_label = y >= 0                      # y == -1 marks rows whose label is unavailable
    labeled_idx = np.where(has_label)[0]
    order = labeled_idx[stratified_order(y[labeled_idx], rng)]
    if test_mask is None:
        test_mask = np.ones(n, dtype=bool)
    if pool_mask is None:
        pool_mask = np.ones(n, dtype=bool)
    test = np.array([i for i in order if test_mask[i]][:n_test])
    in_test = np.zeros(n, dtype=bool)
    in_test[test] = True
    pool = np.array([i for i in range(n) if pool_mask[i] and not in_test[i]])
    pool_lab = pool[has_label[pool]]
    pool_order = pool_lab[stratified_order(y[pool_lab], rng)]
    mean = X[pool].mean(axis=0)
    std = X[pool].std(axis=0) + 1e-6
    return Split(test=test, pool=pool, pool_order=pool_order, mean=mean, std=std)


def standardise(X: np.ndarray, split: Split) -> np.ndarray:
    return ((X - split.mean) / split.std).astype(np.float32)
