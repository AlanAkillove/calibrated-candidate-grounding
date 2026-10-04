"""Shared image-cluster bootstrap for Research Repair v1.

The callback is evaluated once per draw and returns *per-seed effects* for every
named estimate. This lets one sampled image multiset feed every seed, endpoint,
comparison cell, and severity level. Derived quantities such as relative change,
difference-of-differences, and severity macros belong in the callback so they
are formed inside each replicate before seed averaging.

``image_ids`` is aligned with the canonical row table supplied to the callback.
Each draw samples the unique images with replacement and expands every sampled
image to all its rows. Repeated image draws therefore repeat the whole cluster,
including all referring expressions. Point and bootstrap estimates are means of
per-seed effects; they never average predictions across seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["assert_row_alignment", "shared_image_cluster_bootstrap"]

Seed = Hashable
Statistic = Callable[[NDArray[np.int64]], Mapping[str, Mapping[Seed, float]]]


def assert_row_alignment(
    reference: Mapping[str, Sequence[Any] | NDArray[Any]],
    candidate: Mapping[str, Sequence[Any] | NDArray[Any]],
    *,
    required_fields: Sequence[str],
    canonical_id_field: str = "sentence_id",
    reference_name: str = "reference",
    candidate_name: str = "candidate",
) -> NDArray[Any]:
    """Verify canonical row order without joining or deduplicating IDs.

    ``required_fields`` should include every shared per-row anchor available
    in both tables (typically ref/image IDs, split, cell/K, and correctness).
    If the candidate lacks ``canonical_id_field``, the function returns IDs
    from the reference in row order after all anchors match. This is intended
    for legacy CSVs without ``sentence_id``; it never uses ``ref_id`` as a
    unique key.
    """
    fields = tuple(required_fields)
    if not fields:
        raise ValueError("required_fields must name at least one row-alignment anchor")
    if canonical_id_field not in reference:
        raise ValueError(f"{reference_name} has no canonical ID field {canonical_id_field!r}")

    reference_ids = np.asarray(reference[canonical_id_field])
    if reference_ids.ndim != 1:
        raise ValueError(f"{reference_name}.{canonical_id_field} must be one-dimensional")
    n_rows = reference_ids.size
    for field in fields:
        if field not in reference:
            raise ValueError(f"{reference_name} is missing alignment field {field!r}")
        if field not in candidate:
            raise ValueError(f"{candidate_name} is missing alignment field {field!r}")
        expected = np.asarray(reference[field])
        observed = np.asarray(candidate[field])
        if expected.ndim != 1 or observed.ndim != 1:
            raise ValueError(f"alignment field {field!r} must be one-dimensional in both tables")
        if expected.size != n_rows or observed.size != n_rows:
            raise ValueError(
                f"row count differs for {field!r}: {reference_name}={expected.size}, "
                f"{candidate_name}={observed.size}, canonical={n_rows}"
            )
        equal = np.equal(expected, observed)
        if not bool(np.all(equal)):
            first = int(np.flatnonzero(~equal)[0])
            raise ValueError(
                f"row alignment mismatch for {field!r} at canonical row {first}: "
                f"{reference_name}={expected[first]!r}, {candidate_name}={observed[first]!r}"
            )

    if canonical_id_field in candidate:
        observed_ids = np.asarray(candidate[canonical_id_field])
        if observed_ids.ndim != 1 or observed_ids.size != n_rows:
            raise ValueError(f"{candidate_name}.{canonical_id_field} has an incompatible row shape")
        equal = np.equal(reference_ids, observed_ids)
        if not bool(np.all(equal)):
            first = int(np.flatnonzero(~equal)[0])
            raise ValueError(
                f"row alignment mismatch for {canonical_id_field!r} at canonical row {first}: "
                f"{reference_ids[first]!r} != {observed_ids[first]!r}"
            )
    return reference_ids.copy()


def _as_image_ids(image_ids: Sequence[Hashable] | NDArray[Any]) -> tuple[NDArray[Any], list[NDArray[np.int64]]]:
    ids = np.asarray(image_ids)
    if ids.ndim != 1:
        raise ValueError(f"image_ids must be one-dimensional; got shape {ids.shape}")
    if ids.size == 0:
        raise ValueError("image_ids must not be empty")

    # Preserve first-seen order. This makes a fixed seed reproducible against
    # the canonical row order and also supports string-valued image IDs.
    groups: dict[Hashable, list[int]] = {}
    for row, raw_id in enumerate(ids.tolist()):
        if raw_id is None:
            raise ValueError(f"image_ids contains a missing value at row {row}")
        try:
            if bool(np.asarray(raw_id).ndim == 0 and np.asarray(raw_id).dtype.kind == "f" and np.isnan(raw_id)):
                raise ValueError(f"image_ids contains NaN at row {row}")
            groups.setdefault(raw_id, []).append(row)
        except TypeError as exc:
            raise ValueError(f"image_id at row {row} is not hashable: {raw_id!r}") from exc

    if len(groups) < 2:
        raise ValueError("image-cluster bootstrap needs at least two distinct images")
    return ids, [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def _finite_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _summarize_replicates(
    values: NDArray[np.float64], point: float, ci: float, minimum_valid_fraction: float,
) -> dict[str, Any]:
    valid = values[np.isfinite(values)]
    invalid_count = int(values.size - valid.size)
    if valid.size >= 2:
        alpha = (1.0 - ci) / 2.0
        low, high = np.quantile(valid, [alpha, 1.0 - alpha])
        bootstrap_std = float(np.std(valid, ddof=1))
        status = "AVAILABLE_VALID_REPLICATES"
        ci_low: float | None = float(low)
        ci_high: float | None = float(high)
    else:
        bootstrap_std = float("nan")
        status = "UNAVAILABLE_TOO_FEW_VALID_REPLICATES"
        ci_low = None
        ci_high = None
    valid_fraction = float(valid.size / values.size) if values.size else 0.0
    minimum_valid_replicates = int(np.ceil(minimum_valid_fraction * values.size))
    if valid.size == values.size and values.size > 0:
        uncertainty_status = "SUFFICIENT_ALL_PLANNED_REPLICATES_VALID"
        gate_eligible = True
    elif valid.size >= minimum_valid_replicates and valid.size >= 2:
        uncertainty_status = "CONDITIONAL_ON_VALID_DRAWS_NOT_GATE_ELIGIBLE"
        gate_eligible = False
    else:
        uncertainty_status = "UNCERTAINTY_INSUFFICIENT"
        gate_eligible = False
    return {
        "point": float(point),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_std": bootstrap_std,
        "valid_replicates": int(valid.size),
        "invalid_replicates": invalid_count,
        "interval_status": status,
        "uncertainty_status": uncertainty_status,
        "valid_fraction": valid_fraction,
        "minimum_valid_fraction_for_exploratory_ci": float(minimum_valid_fraction),
        "minimum_valid_replicates_for_exploratory_ci": minimum_valid_replicates,
        "gate_eligible": gate_eligible,
        "ci_conditioning": "all_planned_draws_valid" if valid.size == values.size else "conditional_on_finite_draws",
        "replicates": values,
    }


def shared_image_cluster_bootstrap(
    image_ids: Sequence[Hashable] | NDArray[Any],
    statistic: Statistic,
    *,
    point_statistic: Statistic | None = None,
    n_replicates: int = 5000,
    seed: int = 0,
    ci: float = 0.95,
    minimum_valid_fraction: float = 0.95,
) -> dict[str, Any]:
    """Run a shared image-cluster percentile bootstrap across seeds and estimates.

    Parameters
    ----------
    image_ids:
        One image ID per canonical row. All rows belonging to an image are
        sampled together. The row table must be the common cohort for all
        metrics returned by ``statistic``.
    statistic:
        Called once per bootstrap replicate with row indices. It must return
        ``{estimate_name: {seed: per_seed_effect}}``. Return ``NaN`` for a
        mathematically undefined statistic, such as AUROC on a one-class draw
        or a relative effect with a zero denominator. Do not replace it with 0.
        For each estimate, the callback should return the same fixed set of
        seeds on every call. It should calculate each seed's effect first and
        calculate ratios, difference-of-differences, and severity macros within
        that seed and draw. When ``point_statistic`` is omitted, this callback
        is also used for the observed point estimate.
    point_statistic:
        Optional callback used only for the observed point estimate. It allows
        a historical scalar convention to remain at the point while undefined
        bootstrap draws stay NaN and are counted invalid.
    n_replicates, seed, ci:
        Number of image-cluster draws, NumPy generator seed, and percentile CI
        confidence level. The repair protocol uses 5000, 0, and 0.95.

    Returns
    -------
    dict
        ``estimates[name]`` contains the mean-of-seed point estimate, its CI,
        bootstrap standard deviation, valid/invalid draw counts, all raw mean
        replicates, per-seed summaries and raw per-seed replicates, plus the
        standard deviation across the fixed seeds' point estimates. Non-finite
        draws remain NaN in raw arrays and count as invalid. A mean-of-seed
        replicate is valid only when every configured seed is valid.
    """
    ids, groups = _as_image_ids(image_ids)
    n_rep = int(n_replicates)
    if n_rep < 1:
        raise ValueError(f"n_replicates must be >= 1; got {n_replicates}")
    level = float(ci)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError(f"ci must be a confidence level in (0, 1); got {ci}")
    minimum_valid_fraction = float(minimum_valid_fraction)
    if not np.isfinite(minimum_valid_fraction) or not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if not callable(statistic):
        raise TypeError("statistic must be callable")

    def evaluate(callback: Statistic, row_indices: NDArray[np.int64]) -> dict[str, dict[Seed, float]]:
        result = callback(row_indices)
        if not isinstance(result, Mapping) or not result:
            raise ValueError("statistic must return a non-empty mapping of estimates to per-seed effects")
        normalized: dict[str, dict[Seed, float]] = {}
        for raw_name, raw_seed_values in result.items():
            name = str(raw_name)
            if not name:
                raise ValueError("estimate names must not be empty")
            if not isinstance(raw_seed_values, Mapping) or not raw_seed_values:
                raise ValueError(f"estimate {name!r} must map at least one seed to an effect")
            normalized[name] = {seed_key: _finite_float(value) for seed_key, value in raw_seed_values.items()}
        return normalized

    point_rows = np.arange(ids.size, dtype=np.int64)
    point_by_name = evaluate(point_statistic or statistic, point_rows)
    estimate_names = tuple(point_by_name)
    seed_keys_by_name = {name: tuple(point_by_name[name]) for name in estimate_names}
    point_values: dict[str, float] = {}
    aggregate_replicates = {name: np.full(n_rep, np.nan, dtype=np.float64) for name in estimate_names}
    seed_replicates = {
        name: {seed_key: np.full(n_rep, np.nan, dtype=np.float64) for seed_key in seed_keys_by_name[name]}
        for name in estimate_names
    }

    for name in estimate_names:
        values = np.asarray(list(point_by_name[name].values()), dtype=np.float64)
        point_values[name] = float(np.mean(values)) if np.all(np.isfinite(values)) else float("nan")

    rng = np.random.default_rng(int(seed))
    n_images = len(groups)
    for replicate in range(n_rep):
        sampled_clusters = rng.integers(0, n_images, size=n_images)
        sampled_rows = np.concatenate([groups[int(index)] for index in sampled_clusters]).astype(np.int64, copy=False)
        current = evaluate(statistic, sampled_rows)
        if tuple(current) != estimate_names:
            raise ValueError("statistic must return the same estimate names in the same order on every draw")
        for name in estimate_names:
            seed_keys = seed_keys_by_name[name]
            if tuple(current[name]) != seed_keys:
                raise ValueError(f"estimate {name!r} must return the same seed keys in the same order on every draw")
            values = np.asarray([current[name][seed_key] for seed_key in seed_keys], dtype=np.float64)
            for seed_key, value in zip(seed_keys, values, strict=True):
                seed_replicates[name][seed_key][replicate] = value
            if np.all(np.isfinite(values)):
                aggregate_replicates[name][replicate] = float(np.mean(values))

    summaries: dict[str, Any] = {}
    for name in estimate_names:
        per_seed: dict[Seed, Any] = {}
        point_seed_values = np.asarray(list(point_by_name[name].values()), dtype=np.float64)
        seed_sd = (
            float(np.std(point_seed_values, ddof=1))
            if point_seed_values.size >= 2 and np.all(np.isfinite(point_seed_values))
            else float("nan")
        )
        for seed_key in seed_keys_by_name[name]:
            per_seed[seed_key] = _summarize_replicates(
                seed_replicates[name][seed_key], point_by_name[name][seed_key], level, minimum_valid_fraction
            )
        aggregate = _summarize_replicates(
            aggregate_replicates[name], point_values[name], level, minimum_valid_fraction
        )
        summaries[name] = {
            **aggregate,
            "seed_standard_deviation": seed_sd,
            "seed_estimates": dict(point_by_name[name]),
            "per_seed": per_seed,
            "per_seed_replicates": seed_replicates[name],
        }

    return {
        "resample_unit": "image_cluster",
        "n_rows": int(ids.size),
        "n_images": int(n_images),
        "n_replicates": n_rep,
        "seed": int(seed),
        "ci_level": level,
        "method": "percentile",
        "estimates": summaries,
    }
