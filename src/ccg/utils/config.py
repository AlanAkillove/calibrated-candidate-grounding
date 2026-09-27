"""Tiny YAML configuration helper (no dataclass farm, no pydantic).

``configs/phase0.yaml`` declares ``_base_: default.yaml`` and overrides only what
differs, so the frozen defaults live in exactly one file.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

__all__ = ["Config", "load_config", "save_config", "merge_configs", "apply_overrides"]


class Config(MutableMapping):
    """Dict wrapper with dotted access: ``cfg.get("model.hidden_dim", 128)``.

    Reads are tolerant (missing keys return the default), writes are not: a typo
    in ``cfg["..."]`` must fail loudly rather than create a shadow key.
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None, *, source: Optional[str] = None) -> None:
        self._data: Dict[str, Any] = dict(data or {})
        self.source = source

    # -- mapping protocol ----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._resolve(key)

    def __setitem__(self, key: str, value: Any) -> None:
        parts = _split(key)
        node = self._data
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            elif not isinstance(child, dict):
                raise TypeError(f"cannot set {key!r}: {part!r} is a {type(child).__name__}")
            node = child
        node[parts[-1]] = value

    def __delitem__(self, key: str) -> None:
        parts = _split(key)
        node = self._data
        for part in parts[:-1]:
            node = node[part]
        del node[parts[-1]]

    def __iter__(self) -> Iterator[str]:
        return iter(flatten(self._data))

    def __len__(self) -> int:
        return len(flatten(self._data))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self._resolve(key)
        except KeyError:
            return False
        return True

    # -- dotted access -------------------------------------------------------
    def _resolve(self, key: str) -> Any:
        node: Any = self._data
        for part in _split(key):
            if isinstance(node, Mapping):
                if part not in node:
                    raise KeyError(key)
                node = node[part]
            elif isinstance(node, (list, tuple)):
                try:
                    node = node[int(part)]
                except (ValueError, IndexError) as exc:
                    raise KeyError(key) from exc
            else:
                raise KeyError(key)
        return node

    def get(self, key: str, default: Any = None) -> Any:
        """Dotted ``get`` with a default (``cfg.get("metrics.bins", 15)``)."""
        try:
            return self._resolve(key)
        except KeyError:
            return default

    def require(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None:
            raise KeyError(f"required config key {key!r} is missing (source={self.source})")
        return value

    def set(self, key: str, value: Any) -> "Config":
        self[key] = value
        return self

    # -- composition ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def merged(self, other: Optional[Mapping[str, Any]]) -> "Config":
        """Return a new config with ``other`` deep-merged on top of this one."""
        return Config(merge_configs(self._data, other or {}), source=self.source)

    def without_base(self) -> "Config":
        self._data.pop("_base_", None)
        return self

    def keys(self):  # noqa: D401 - flat dotted keys, matching __iter__
        return list(iter(self))

    def __repr__(self) -> str:
        return f"Config(source={self.source!r}, keys={len(self)})"


def _split(key: str) -> list[str]:
    if not isinstance(key, str):
        raise TypeError(f"config keys must be str, got {type(key)}")
    return [part for part in key.split(".") if part]


def flatten(data: Mapping[str, Any], prefix: str = "") -> Iterable[str]:
    """Yield dotted keys of a nested mapping (leaf lists count as one key)."""
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            yield from flatten(value, prefix=f"{full}.")
        else:
            yield full


def merge_configs(base: Optional[Mapping[str, Any]], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recursive dict merge; mappings merge, everything else (lists included) replaces."""
    result: Dict[str, Any] = copy.deepcopy(dict(base or {}))
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: str | Path,
    *,
    follow_base: bool = True,
    _seen: Optional[set] = None,
) -> Config:
    """Load a YAML file into a :class:`Config`, resolving ``_base_`` inheritance.

    ``_base_: default.yaml`` is resolved relative to the *including* file, and a
    cycle raises instead of recursing.  ``yaml`` is imported lazily.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file {path} not found")
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(raw)}")
    seen = _seen or set()
    seen.add(str(path.resolve()))
    base_name = raw.pop("_base_", None)
    if follow_base and base_name:
        if not isinstance(base_name, str):
            raise TypeError(f"{path}: _base_ must be a str, got {type(base_name)}")
        base_path = (path.parent / base_name).resolve()
        if str(base_path) in seen:
            raise ValueError(f"cyclic _base_ chain reaching {base_path}")
        base = load_config(base_path, follow_base=True, _seen=seen)
        merged = merge_configs(base.to_dict(), raw)
    else:
        merged = raw
    return Config(merged, source=str(path))


def save_config(config: Mapping[str, Any] | Config, path: str | Path) -> Path:
    """Write a config back to YAML (deterministic key order, no flow style)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = _require_yaml()
    data = config.to_dict() if isinstance(config, Config) else dict(config)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True, allow_unicode=True, default_flow_style=False)
    return path


def apply_overrides(config: Config, overrides: Iterable[str]) -> Config:
    """Apply CLI ``key=value`` / ``key.sub=value`` overrides (values are yaml-parsed)."""
    yaml = _require_yaml()
    out = Config(config.to_dict(), source=config.source)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key=value")
        key, _, text = item.partition("=")
        out[key.strip()] = yaml.safe_load(text)
    return out


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("YAML config support needs pyyaml (declared in pyproject.toml)") from exc
    return yaml
