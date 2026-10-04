"""Tie-aware pair accounting on frozen predictions, using the historical S/F/E names.

S = correct at both K; F = flipped from correct to wrong; E = wrong at both K.
The terms below are metric identities, not identified causal contributions.
"""
from __future__ import annotations

import numpy as np


def _weighted(coefficient: float, value: float) -> float:
    return 0.0 if coefficient == 0.0 else coefficient * value


def derive_pair_terms(n_s: float, n_f: float, n_e: float,
                      se: float, fe: float, sf: float) -> dict[str, float]:
    """Express the label path as beta*(SF-SE) + alpha*(SE-FE)."""
    alpha = n_f / (n_s + n_f) if n_s + n_f else float("nan")
    beta = n_f / (n_f + n_e) if n_f + n_e else float("nan")
    a5 = _weighted(1.0 - alpha, se) + _weighted(alpha, fe)
    a50 = _weighted(1.0 - beta, se) + _weighted(beta, sf)
    new_comparison = _weighted(beta, sf - se)
    removed_comparison = _weighted(alpha, se - fe)
    return {"alpha": alpha, "beta": beta, "A_r5": a5, "A_r50": a50,
            "new_comparison": new_comparison, "removed_comparison": removed_comparison,
            "label_path": a50 - a5}


class FrozenPairAccounting:
    """Precompute exact score tie groups; resampling only changes row multiplicities.

    Weighted pair wins equal explicit duplicated-row AUROC. This avoids sorting
    10,000 rows afresh in each of the 5,000 image-cluster bootstrap draws.
    """

    def __init__(self, p5, r5, p50, r50):
        vectors = [np.asarray(x, dtype=float) for x in (p5, r5, p50, r50)]
        if any(x.ndim != 1 for x in vectors) or not vectors[0].size:
            raise ValueError("inputs must be nonempty vectors")
        self.n = vectors[0].size
        if any(x.size != self.n or not np.isfinite(x).all() for x in vectors):
            raise ValueError("inputs must be finite aligned vectors")
        p5, r5, p50, r50 = vectors
        if any(not np.isin(x, [0, 1]).all() for x in (r5, r50)):
            raise ValueError("correctness must be binary")
        if np.any(r50 > r5):
            raise ValueError("G is nonempty; nested unique-target assumption failed")
        self.groups = np.where(r50 == 1, 0, np.where(r5 == 1, 1, 2))
        self.tables = {}
        for tag, scores in (("p5", p5), ("p50", p50)):
            levels, inverse = np.unique(scores, return_inverse=True)
            self.tables[tag] = (levels.size, self.groups * levels.size + inverse)

    def evaluate(self, weights=None) -> dict[str, float]:
        w = np.ones(self.n) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (self.n,) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError("weights must be finite nonnegative aligned row multiplicities")
        ns, nf, ne = np.bincount(self.groups, weights=w, minlength=3)
        result = {"nS": float(ns), "nF": float(nf), "nE": float(ne)}
        terms = {}
        for tag, (n_levels, indices) in self.tables.items():
            mass = np.bincount(indices, weights=w, minlength=3*n_levels).reshape(3, n_levels)
            lower = np.cumsum(mass, axis=1) - mass
            pairs = {}
            for name, a, b in (("SE", 0, 2), ("FE", 1, 2), ("SF", 0, 1)):
                denominator = mass[a].sum() * mass[b].sum()
                pairs[name] = (float(np.dot(mass[a], lower[b] + 0.5*mass[b]) / denominator)
                               if denominator else float("nan"))
                result[f"U_{name}_{tag}"] = pairs[name]
            term = derive_pair_terms(ns, nf, ne, pairs["SE"], pairs["FE"], pairs["SF"])
            terms[tag] = term
            for name in ("new_comparison", "removed_comparison", "label_path"):
                result[f"{name}_{tag}"] = term[name]
        result.update(alpha=terms["p5"]["alpha"], beta=terms["p5"]["beta"],
                      A00=terms["p5"]["A_r5"], A10=terms["p5"]["A_r50"],
                      A01=terms["p50"]["A_r5"], A11=terms["p50"]["A_r50"])
        for pair in ("SE", "FE", "SF"):
            result[f"delta_U_{pair}"] = result[f"U_{pair}_p50"] - result[f"U_{pair}_p5"]
        alpha, beta = result["alpha"], result["beta"]
        result["C0_SE"] = _weighted(1-alpha, result["delta_U_SE"])
        result["C0_FE"] = _weighted(alpha, result["delta_U_FE"])
        result["C1_SE"] = _weighted(1-beta, result["delta_U_SE"])
        result["C1_SF"] = _weighted(beta, result["delta_U_SF"])
        result["C0"] = result["A01"] - result["A00"]
        result["C1"] = result["A11"] - result["A10"]
        result["I_new_comparison"] = result["new_comparison_p50"] - result["new_comparison_p5"]
        result["I_removed_comparison"] = result["removed_comparison_p50"] - result["removed_comparison_p5"]
        result["interaction_I"] = result["label_path_p50"] - result["label_path_p5"]
        return result
