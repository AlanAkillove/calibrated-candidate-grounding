"""Load-and-predict access to the frozen RefCOCO+ reliability models (protocol A11.7).

Protocol amendment A11 forbids *any* fitting on the external production path
(instruction sections 8/10): the RefCOCOg runner must

    load saved coefficients -> predict

and must never reconstruct those coefficients by re-running lbfgs.  The
pre-existing :func:`ccg.semantic.frozen.recover_frozen_models` cannot be used on
that path because it re-fits both logistic models as part of its A8.4
verification, and A10 established that its lbfgs output is sensitive to the
BLAS thread count at the ``1e-9`` level.

This module therefore holds the *inference-only* half of the frozen pipeline:

* :class:`FrozenSeedArtifacts` / :class:`FrozenExternalModels` - the coefficient
  bundle (per B3 seed: 17-d Stats weights, 33-d E1b weights, both intercepts, the
  two train-only normalisations, the temperature, the frozen feature orderings);
* :func:`save_models` / :func:`load_models` - the on-disk round trip
  (``models.json`` + ``models.sha256``), with a schema check on load;
* :func:`verify_against_frozen_artifacts` - the bundle re-checked against the
  RefCOCO+ artifacts it came from (Phase 1 ``e1_logistic/coefficients.csv``,
  Phase 0.5 ``{scorer}_normalisation.json``, ``selection.json`` temperatures and
  the Phase 1F stored predictions), all under the untouched ``_TOL`` of A8.4;
* :meth:`FrozenExternalModels.predict` - ``normalize_apply`` plus a closed-form
  ``sigmoid(x @ coef + b)``, i.e. exactly :func:`ccg.semantic.frozen.apply_frozen`
  without sklearn.

Nothing in this module imports sklearn or calls ``fit``; :func:`assert_no_fit_path`
is exported so the test suite can pin that property down statically, and
:func:`fit_is_forbidden` is a runtime tripwire the runner installs around its own
inference stage.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..reliability import features as rfeat
from ..reliability import models as rmodels
from . import features as sfeat
from . import frozen as sfrozen

__all__ = [
    "ARTIFACT_SCHEMA",
    "FORBIDDEN_PRODUCTION_CALLS",
    "FrozenSeedArtifacts",
    "FrozenExternalModels",
    "assert_no_fit_path",
    "default_artifact_root",
    "fit_is_forbidden",
    "load_models",
    "save_models",
    "verify_against_frozen_artifacts",
]

#: Schema tag written into (and required from) every exported bundle.
ARTIFACT_SCHEMA = "a11-frozen-reliability-v1"
#: Call names that must never appear on the external production path (§8).
FORBIDDEN_PRODUCTION_CALLS: Tuple[str, ...] = (
    ".fit(",
    ".fit_transform(",
    "fit_predict(",
    "LogisticRegression(",
    "recover_frozen_models(",
    "calibrate(",
)
#: Default location of the exported RefCOCO+ coefficient bundle.
DEFAULT_ARTIFACT_ROOT = Path("results/phase1e_refcocog_external/frozen_models")
#: Relative name of the bundle file inside that root.
BUNDLE_FILENAME = "models.json"
#: Relative name of the checksum file inside that root.
CHECKSUM_FILENAME = "models.sha256"
#: Tolerance of the coefficient / prediction round trip (A8.4's own bar).
_TOL = sfrozen._TOL


def default_artifact_root() -> Path:
    """The repository-relative root the A11 runner reads its coefficients from."""
    return DEFAULT_ARTIFACT_ROOT


# ---------------------------------------------------------------------------
# the closed-form scorer (no sklearn, no fitting)
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid of a float64 vector."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.empty_like(values)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


@dataclass(frozen=True)
class FrozenSeedArtifacts:
    """One B3 seed's frozen reliability stack: weights, intercepts, normalisations.

    ``stats_coef`` is the 17-d Stats Logistic weight vector, ``e1b_coef`` the
    33-d E1b vector (the frozen A5 ordering: 17 score statistics followed by the
    16 semantic statistics), and the two ``*_fit`` objects are the RefCOCO+
    *train-only* standardisations those models were fitted on.
    """

    scorer: str
    temperature: float
    stats_coef: np.ndarray
    stats_intercept: float
    e1b_coef: np.ndarray
    e1b_intercept: float
    stats_C: float
    e1b_C: float
    stats_fit: rfeat.NormalizationFit
    sem_fit: rfeat.NormalizationFit

    def __post_init__(self) -> None:
        object.__setattr__(self, "scorer", str(self.scorer))
        for name in ("stats_coef", "e1b_coef"):
            values = np.ascontiguousarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if values.size == 0 or not bool(np.all(np.isfinite(values))):
                raise ValueError(f"{self.scorer}: {name} must be a non-empty finite vector")
            object.__setattr__(self, name, values)
        for name in ("temperature", "stats_intercept", "e1b_intercept", "stats_C", "e1b_C"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{self.scorer}: {name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)
        if self.stats_coef.size != len(rfeat.stat_feature_names()):
            raise ValueError(
                f"{self.scorer}: stats_coef has {self.stats_coef.size} entries, expected "
                f"{len(rfeat.stat_feature_names())} (the frozen 17-d score statistics)"
            )
        expected_e1b = self.stats_coef.size + len(sfeat.SEMANTIC_STAT_NAMES)
        if self.e1b_coef.size != expected_e1b:
            raise ValueError(
                f"{self.scorer}: e1b_coef has {self.e1b_coef.size} entries, expected "
                f"{expected_e1b} (17 stats + 16 semantic)"
            )

    # -- inference -----------------------------------------------------------
    def stats_probability(self, stats17_std: np.ndarray) -> np.ndarray:
        """``P(correct)`` of the frozen Stats Logistic on standardised rows."""
        return self._apply(self.stats_coef, self.stats_intercept, stats17_std, "stats17")

    def e1b_probability(self, e1b_std: np.ndarray) -> np.ndarray:
        """``P(correct)`` of the frozen E1b on standardised 33-d rows."""
        return self._apply(self.e1b_coef, self.e1b_intercept, e1b_std, "e1b")

    def _apply(
        self, coef: np.ndarray, intercept: float, x: np.ndarray, label: str
    ) -> np.ndarray:
        rows = np.asarray(x, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != coef.size:
            raise ValueError(
                f"{self.scorer}: {label} input must be [n, {coef.size}], got {rows.shape}"
            )
        if rows.size and not bool(np.all(np.isfinite(rows))):
            raise ValueError(f"{self.scorer}: {label} input holds non-finite values")
        return _sigmoid(rows @ coef + intercept)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorer": self.scorer,
            "temperature": self.temperature,
            "stats_coef": self.stats_coef.tolist(),
            "stats_intercept": self.stats_intercept,
            "e1b_coef": self.e1b_coef.tolist(),
            "e1b_intercept": self.e1b_intercept,
            "stats_C": self.stats_C,
            "e1b_C": self.e1b_C,
            "stats_fit": self.stats_fit.to_dict(),
            "sem_fit": self.sem_fit.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenSeedArtifacts":
        return cls(
            scorer=str(payload["scorer"]),
            temperature=float(payload["temperature"]),
            stats_coef=np.asarray(payload["stats_coef"], dtype=np.float64),
            stats_intercept=float(payload["stats_intercept"]),
            e1b_coef=np.asarray(payload["e1b_coef"], dtype=np.float64),
            e1b_intercept=float(payload["e1b_intercept"]),
            stats_C=float(payload["stats_C"]),
            e1b_C=float(payload["e1b_C"]),
            stats_fit=rfeat.NormalizationFit.from_dict(payload["stats_fit"]),
            sem_fit=rfeat.NormalizationFit.from_dict(payload["sem_fit"]),
        )


@dataclass(frozen=True)
class FrozenExternalModels:
    """The whole exported bundle (three B3 seeds) plus its frozen feature contract."""

    seeds: Mapping[str, FrozenSeedArtifacts]
    stats_feature_names: Tuple[str, ...]
    semantic_feature_names: Tuple[str, ...]
    source: Mapping[str, Any]

    def __post_init__(self) -> None:
        seeds = {str(key): value for key, value in dict(self.seeds).items()}
        if not seeds:
            raise ValueError("FrozenExternalModels needs at least one seed")
        for scorer, artifact in seeds.items():
            if not isinstance(artifact, FrozenSeedArtifacts):
                raise TypeError(f"{scorer}: bundle entries must be FrozenSeedArtifacts")
            if artifact.scorer != scorer:
                raise ValueError(f"bundle key {scorer!r} != payload scorer {artifact.scorer!r}")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(
            self, "stats_feature_names", tuple(str(name) for name in self.stats_feature_names)
        )
        object.__setattr__(
            self,
            "semantic_feature_names",
            tuple(str(name) for name in self.semantic_feature_names),
        )
        object.__setattr__(self, "source", dict(self.source))

    @property
    def scorers(self) -> Tuple[str, ...]:
        """Seed model names in a deterministic (sorted) order."""
        return tuple(sorted(self.seeds))

    # -- inference (the only forward path the A11 runner may use) ------------
    def seed(self, scorer: str) -> FrozenSeedArtifacts:
        try:
            return self.seeds[str(scorer)]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"{scorer!r} is not in the frozen bundle ({sorted(self.seeds)})"
            ) from exc

    def predict(
        self,
        scorer: str,
        stats17_raw: np.ndarray,
        sem16_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load-and-predict twin of :func:`ccg.semantic.frozen.apply_frozen`.

        ``stats17_raw`` / ``sem16_raw`` are the *un-standardised* ``[n, 17]`` and
        ``[n, 16]`` feature blocks of the external rows; the stored RefCOCO+
        train-only normalisations are applied as-is (never re-estimated here).
        Returns ``(stats_conf, e1b_conf)``, both ``[n]`` float64 ``P(correct)``.
        """
        artifact = self.seed(scorer)
        stats_std = rfeat.normalize_apply(np.asarray(stats17_raw, dtype=np.float64), artifact.stats_fit)
        sem_std = rfeat.normalize_apply(np.asarray(sem16_raw, dtype=np.float64), artifact.sem_fit)
        stats_conf = artifact.stats_probability(stats_std)
        e1b_conf = artifact.e1b_probability(np.hstack([stats_std, sem_std]))
        return stats_conf, e1b_conf

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA,
            "stats_feature_names": list(self.stats_feature_names),
            "semantic_feature_names": list(self.semantic_feature_names),
            "seeds": {scorer: self.seeds[scorer].to_dict() for scorer in self.scorers},
            "source": dict(self.source),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenExternalModels":
        version = payload.get("schema_version")
        if version != ARTIFACT_SCHEMA:
            raise ValueError(
                f"{BUNDLE_FILENAME}: schema_version {version!r} does not match {ARTIFACT_SCHEMA!r}"
            )
        seeds = {
            str(scorer): FrozenSeedArtifacts.from_dict(dict(bundle))
            for scorer, bundle in dict(payload["seeds"]).items()
        }
        bundle = cls(
            seeds=seeds,
            stats_feature_names=tuple(payload["stats_feature_names"]),
            semantic_feature_names=tuple(payload["semantic_feature_names"]),
            source=dict(payload.get("source") or {}),
        )
        bundle.check_feature_contract()
        return bundle

    def check_feature_contract(self) -> None:
        """The bundle's feature names must be the frozen ones, in the frozen order."""
        want_stats = tuple(rfeat.stat_feature_names())
        want_sem = tuple(sfeat.SEMANTIC_STAT_NAMES)
        if self.stats_feature_names != want_stats:
            raise ValueError(
                "bundle stats feature ordering drifted from the frozen 17-d definition"
            )
        if self.semantic_feature_names != want_sem:
            raise ValueError(
                "bundle semantic feature ordering drifted from the frozen 16-d definition"
            )
        for scorer, artifact in self.seeds.items():
            if tuple(artifact.stats_fit.keys or ()) and tuple(artifact.stats_fit.keys) != want_stats:
                raise ValueError(f"{scorer}: stats_fit keys are not the frozen 17-d order")
            if tuple(artifact.sem_fit.keys or ()) and tuple(artifact.sem_fit.keys) != want_sem:
                raise ValueError(f"{scorer}: sem_fit keys are not the frozen 16-d order")


# ---------------------------------------------------------------------------
# on-disk round trip
# ---------------------------------------------------------------------------
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_models(root: Path | str, bundle: FrozenExternalModels) -> Dict[str, Any]:
    """Write ``root/models.json`` + ``root/models.sha256`` atomically; return provenance."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload = bundle.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    target = root / BUNDLE_FILENAME
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    import os

    os.replace(tmp, target)
    (root / CHECKSUM_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": ARTIFACT_SCHEMA,
                BUNDLE_FILENAME: digest,
                "sha256_of_file": _file_sha256(target),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "root": str(root),
        "bundle_sha256": digest,
        "bundle_file_sha256": _file_sha256(target),
        "n_seeds": len(bundle.seeds),
        "seeds": list(bundle.scorers),
    }


def load_models(root: Path | str, *, verify_checksum: bool = True) -> FrozenExternalModels:
    """Read the exported bundle back; optionally pin it to its recorded checksum."""
    root = Path(root)
    target = root / BUNDLE_FILENAME
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found - export it first with "
            "tools/freeze_a11_reliability_artifacts.py (RefCOCO+ only)"
        )
    text = target.read_text(encoding="utf-8")
    if verify_checksum:
        sidecar = root / CHECKSUM_FILENAME
        if not sidecar.exists():
            raise FileNotFoundError(f"{sidecar} not found - the bundle must carry its checksum")
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        if recorded.get(BUNDLE_FILENAME) != hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise AssertionError(
                f"{target}: content does not match {CHECKSUM_FILENAME} - the frozen "
                "coefficient bundle was modified after export"
            )
    return FrozenExternalModels.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# bundle-level verification against the RefCOCO+ artifacts it came from
# ---------------------------------------------------------------------------
def _load_e1b_reference(phase1_root: Path) -> Dict[str, Tuple[np.ndarray, Optional[float]]]:
    """``scorer -> (33-d coefficients, intercept)`` from Phase 1 ``coefficients.csv``.

    The intercept stays optional per scorer: the frozen table carries only the
    coefficient column for these models (no ``intercept`` row), so the lookup is
    keyed on the coefficient vectors and the intercept is fetched with ``.get``.
    Zipping the two dictionaries would silently drop every scorer whenever the
    intercept block is empty.
    """
    coefs, intercepts = sfrozen._load_e1b_coefficients(Path(phase1_root))
    return {scorer: (coef, intercepts.get(scorer)) for scorer, coef in coefs.items()}


def _load_phase1f_scores(path: Path, scorer: str) -> Dict[str, np.ndarray]:
    """``cell -> stored arrays`` of one Phase 1F prediction file (RefCOCO+ rows)."""
    out: Dict[str, np.ndarray] = {}
    for cell in ("rand5", "hard5"):
        target = Path(path) / "predictions" / f"{cell}__{scorer}.npz"
        if not target.exists():
            raise FileNotFoundError(f"{target} not found - run Phase 1F first")
        with np.load(target, allow_pickle=False) as store:
            out[cell] = {
                "scores": np.asarray(store["scores"], dtype=np.float32),
                "stats": np.asarray(store["conf_stats"], dtype=np.float64),
                "e1b": np.asarray(store["conf_e1b"], dtype=np.float64),
            }
    return out


def verify_against_frozen_artifacts(
    bundle: FrozenExternalModels,
    *,
    phase05_root: Path | str = sfrozen.DEFAULT_PHASE05,
    phase1_root: Path | str = sfrozen.DEFAULT_PHASE1,
    phase1f_root: Path | str = Path("results/phase1f_hard_semantic"),
    b3_root: Path | str = sfrozen.DEFAULT_B3_ROOT,
    features_dir: Optional[Path | str] = None,
    scorers: Optional[Sequence[str]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Re-check the exported coefficients against the RefCOCO+ artifacts they cite.

    Four independent comparisons, every one under
    :data:`ccg.semantic.frozen._TOL` (the A8.4 bar, unchanged by A11):

    1. the 33-d E1b weight vector against ``e1_logistic/coefficients.csv``;
    2. the Stats-17 train-only normalisation against
       ``phase05/features/{scorer}_normalisation.json``;
    3. the temperature against ``phase0b_independent/seed_{s}/eval_metadata.json``;
    4. the Stats prediction on the **stored RefCOCO+ Phase 1F rows** - its raw B3
       logits are on disk, so the whole ``stat_features -> normalise -> sigmoid``
       chain is re-run from those logits and compared with ``conf_stats``.

    The E1b side of the same question (which additionally needs the stored CLIP
    embeddings) is answered by the A11 anchor-recovery stage, and the
    sklearn-versus-closed-form equivalence of this module's arithmetic is proved
    once by ``tools/freeze_a11_reliability_artifacts.py``.

    Raises :class:`AssertionError` on the first drift; returns the per-seed maxima.
    """
    emit = log or (lambda message: None)
    phase05_root = Path(phase05_root)
    phase1_root = Path(phase1_root)
    phase1f_root = Path(phase1f_root)
    features_dir = Path(features_dir) if features_dir is not None else phase05_root / "features"
    e1b_reference = _load_e1b_reference(phase1_root)
    wanted = tuple(scorers) if scorers else bundle.scorers
    per_seed: Dict[str, Dict[str, float]] = {}

    for scorer in wanted:
        artifact = bundle.seed(scorer)
        deltas: Dict[str, float] = {}
        # 1: E1b weights vs the persisted Phase 1 coefficient table.
        reference = e1b_reference.get(scorer)
        if reference is None:
            raise AssertionError(f"{scorer}: no E1b coefficients in e1_logistic/coefficients.csv")
        ref_coef, ref_intercept = reference
        deltas["e1b_coefficients_max_abs"] = float(np.abs(artifact.e1b_coef - ref_coef).max())
        if ref_intercept is not None:
            deltas["e1b_intercept_max_abs"] = float(abs(artifact.e1b_intercept - ref_intercept))
        # 2: the frozen train-only Stats-17 normalisation.
        ref_mean, ref_std = sfrozen._load_stats17_normalisation(features_dir, scorer)
        deltas["stats17_normalisation_max_abs"] = max(
            float(np.abs(np.asarray(artifact.stats_fit.mean) - ref_mean).max()),
            float(np.abs(np.asarray(artifact.stats_fit.std) - ref_std).max()),
        )
        # 3: the temperature the score statistics were built with.
        ref_temperature = sfrozen._load_temperature(Path(b3_root), scorer)
        deltas["temperature_max_abs"] = float(abs(artifact.temperature - float(ref_temperature)))
        # 4: the load-and-predict Stats path on the stored RefCOCO+ rows.
        pred_delta = 0.0
        for cell, arrays in _load_phase1f_scores(phase1f_root, scorer).items():
            scores64 = arrays["scores"].astype(np.float64)
            stats17 = np.asarray(
                rfeat.stat_features(scores64, temperature=artifact.temperature), dtype=np.float64
            )
            stats_conf = artifact.stats_probability(
                rfeat.normalize_apply(stats17, artifact.stats_fit)
            )
            pred_delta = max(pred_delta, float(np.abs(stats_conf - arrays["stats"]).max()))
        deltas["stats_logistic_pred_max_abs"] = pred_delta
        for name, value in deltas.items():
            if not (value <= _TOL):
                raise AssertionError(
                    f"{scorer}: A11 bundle check {name} = {value:.3e} exceeds tolerance {_TOL:.0e}"
                )
        per_seed[scorer] = {name: float(value) for name, value in deltas.items()}
        emit(
            f"[a11-bundle] {scorer}: verified "
            + ", ".join(f"{name}={per_seed[scorer][name]:.2e}" for name in sorted(per_seed[scorer]))
        )

    aggregate = {
        name: float(max(per_seed[scorer][name] for scorer in per_seed))
        for name in sorted({key for scorer in per_seed.values() for key in scorer})
    }
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "tolerance": float(_TOL),
        "scorers": list(wanted),
        "checks": per_seed,
        "all": aggregate,
    }


# ---------------------------------------------------------------------------
# the no-fit guarantees (§8)
# ---------------------------------------------------------------------------
def _code_only_lines(path: Path) -> List[str]:
    """``path`` as one line per source line, with string and comment tokens removed.

    Detection of the forbidden call names must not be confused by prose, so this
    rebuilds each line from the token stream while dropping ``STRING`` and
    ``COMMENT`` tokens (docstrings included).  Attribute access and call
    punctuation are kept verbatim, so ``model.fit(`` survives as ``.fit(``.
    """
    import io
    import tokenize

    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    blanked = [""] * len(lines)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except tokenize.TokenError as exc:  # pragma: no cover - malformed source
        raise ValueError(f"{path}: cannot tokenise the source: {exc}") from exc
    for token in tokens:
        if token.type in (tokenize.STRING, tokenize.COMMENT):
            for row in range(token.start[0], token.end[0] + 1):
                if 1 <= row <= len(blanked):
                    blanked[row - 1] += " "
            continue
        if token.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            continue
        row = token.start[0]
        if 1 <= row <= len(blanked):
            blanked[row - 1] += token.string
    return blanked


def assert_no_fit_path(
    paths: Sequence[Path | str], *, extra: Sequence[str] = ()
) -> Dict[str, Any]:
    """Statically reject fitting call sites on the external production path (§8).

    Raises :class:`AssertionError` naming the first offender when any of
    :data:`FORBIDDEN_PRODUCTION_CALLS` (plus ``extra``) appears in the executable
    code of ``paths``; returns the scanned file list and the needle set otherwise.
    """
    needles = list(FORBIDDEN_PRODUCTION_CALLS) + [str(item) for item in extra]
    offenders: List[str] = []
    scanned: List[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        scanned.append(str(path))
        for line_no, line in enumerate(_code_only_lines(path), start=1):
            for needle in needles:
                if needle in line:
                    offenders.append(f"{path}:{line_no}: {needle} :: {line.strip()[:80]}")
    if offenders:
        raise AssertionError(
            "the external production path must load frozen coefficients only; "
            "forbidden call(s) found: " + " | ".join(offenders[:8])
        )
    return {"files": scanned, "needles": needles, "n_files": len(scanned)}


class _FitTripwire:
    """Context manager that turns any fitted-model entry point into an error."""

    def __init__(self) -> None:
        self._saved: List[Tuple[Any, str, Any]] = []
        self.hits: List[str] = []

    def __enter__(self) -> "_FitTripwire":
        targets = [(rmodels.LogisticModel, "fit"), (rmodels.LogisticModel, "fit_transform")]
        for owner, attribute in targets:
            original = getattr(owner, attribute, None)
            if original is None:
                continue

            def _blocked(*args: Any, _owner: Any = owner, _name: str = attribute, **kwargs: Any) -> Any:
                self.hits.append(f"{_owner.__name__}.{_name}")
                raise AssertionError(
                    f"A11 external path attempted {_owner.__name__}.{_name}() - "
                    "frozen coefficients must be loaded, never re-fitted (protocol A11.7)"
                )

            setattr(owner, attribute, _blocked)
            self._saved.append((owner, attribute, original))
        try:  # sklearn's own entry point, as a second net
            from sklearn.linear_model._logistic import LogisticRegression

            original = LogisticRegression.fit

            def _blocked_sklearn(*args: Any, **kwargs: Any) -> Any:
                self.hits.append("LogisticRegression.fit")
                raise AssertionError(
                    "A11 external path attempted LogisticRegression.fit() - forbidden by A11.7"
                )

            LogisticRegression.fit = _blocked_sklearn
            self._saved.append((LogisticRegression, "fit", original))
        except Exception:  # pragma: no cover - sklearn is a hard dependency
            pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for owner, attribute, original in reversed(self._saved):
            setattr(owner, attribute, original)
        self._saved = []


def fit_is_forbidden() -> _FitTripwire:
    """Runtime guard: ``with fit_is_forbidden(): ...`` makes any refit raise."""
    return _FitTripwire()
