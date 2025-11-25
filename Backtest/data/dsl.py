"""DSL loader for factor pre-computation."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import yaml

# Ensure the Factor Evaluation directory is importable so we can leverage the existing DSL helpers.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTOR_EVAL_DIR = PROJECT_ROOT / "Factor Evaluation"
if FACTOR_EVAL_DIR.exists() and str(FACTOR_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(FACTOR_EVAL_DIR))

try:  # pragma: no cover - optional dependency
    from step3_factor_engine_04 import DEFAULT_WINDOW_NA_THRESH, DSLParseError, evaluate_expression, parse_dsl
except Exception:  # noqa: BLE001
    DEFAULT_WINDOW_NA_THRESH = 0.2

    class DSLParseError(ValueError):
        """Fallback DSL exception when step3 module is unavailable."""

    def parse_dsl(source: str):  # type: ignore[redef]
        raise DSLParseError("step3_factor_engine_04.parse_dsl is required for expression-based factors.")

    def evaluate_expression(parsed, data: pd.DataFrame, na_thresh: float):  # type: ignore[redef]
        raise DSLParseError("step3_factor_engine_04.evaluate_expression is required for expression-based factors.")


class FactorComputationError(RuntimeError):
    """Raised when a factor definition cannot be evaluated."""


def _load_structured_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    if path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported DSL format for '{path}'. Use .yaml, .yml or .json.")


def _resolve_callable(spec: str):
    if ":" in spec:
        module_path, attr = spec.split(":", 1)
    else:
        module_path, attr = spec.rsplit(".", 1)
    module = importlib.import_module(module_path)
    target = module
    for part in attr.split("."):
        target = getattr(target, part)
    return target


@dataclass
class FactorSpec:
    name: str
    type: str = "expression"
    expression: Optional[str] = None
    source: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    fillna: Optional[float] = None
    na_threshold: Optional[float] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FactorSpec":
        if "name" not in payload:
            raise ValueError("Factor specification missing 'name'.")
        return cls(
            name=str(payload["name"]),
            type=str(payload.get("type", "expression")).lower(),
            expression=payload.get("expression") or payload.get("formula"),
            source=payload.get("source"),
            params=dict(payload.get("params", {})),
            fillna=payload.get("fillna"),
            na_threshold=payload.get("na_threshold"),
        )


class FactorRecipe:
    def __init__(self, specs: Iterable[FactorSpec], *, default_na_threshold: float = DEFAULT_WINDOW_NA_THRESH) -> None:
        specs = list(specs)
        if not specs:
            raise ValueError("Factor recipe is empty; define at least one factor.")
        self.specs = specs
        self.default_na_threshold = default_na_threshold

    @classmethod
    def from_file(cls, path: Path) -> "FactorRecipe":
        payload = _load_structured_file(path)
        factors = payload.get("factors")
        if not isinstance(factors, list):
            raise ValueError("DSL payload must contain a 'factors' list.")
        specs = [FactorSpec.from_dict(item) for item in factors]
        default_na_threshold = float(payload.get("default_na_threshold", DEFAULT_WINDOW_NA_THRESH))
        return cls(specs, default_na_threshold=default_na_threshold)

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        results: Dict[str, pd.Series] = {}
        for spec in self.specs:
            if spec.type == "expression":
                results[spec.name] = self._compute_expression(spec, data)
            elif spec.type in {"library_function", "callable"}:
                results.update(self._compute_library_function(spec, data))
            else:
                raise FactorComputationError(f"Unsupported factor type '{spec.type}' for factor '{spec.name}'.")
        frame = pd.DataFrame(results, index=data.index)
        frame.sort_index(inplace=True)
        return frame

    def _compute_expression(self, spec: FactorSpec, data: pd.DataFrame) -> pd.Series:
        if not spec.expression:
            raise FactorComputationError(f"Expression factor '{spec.name}' missing 'expression' definition.")
        dsl_target = spec.name.upper()
        dsl_text = f"{dsl_target} = {spec.expression}"
        parsed = parse_dsl(dsl_text)
        na_threshold = float(spec.na_threshold if spec.na_threshold is not None else self.default_na_threshold)
        series = evaluate_expression(parsed, data, na_threshold)
        series = series.rename(spec.name).reindex(data.index)
        if spec.fillna is not None:
            series = series.fillna(spec.fillna)
        return series

    def _compute_library_function(self, spec: FactorSpec, data: pd.DataFrame) -> Dict[str, pd.Series]:
        if not spec.source:
            raise FactorComputationError(f"Library factor '{spec.name}' missing 'source'.")
        func = _resolve_callable(spec.source)
        result = func(data, **spec.params)
        if isinstance(result, pd.Series):
            series = result.rename(spec.name).reindex(data.index)
            if spec.fillna is not None:
                series = series.fillna(spec.fillna)
            return {spec.name: series}
        if isinstance(result, pd.DataFrame):
            renamed = {}
            for column in result.columns:
                key = f"{spec.name}_{column}" if column != spec.name else column
                series = result[column].reindex(data.index)
                if spec.fillna is not None:
                    series = series.fillna(spec.fillna)
                renamed[key] = series
            return renamed
        raise FactorComputationError(
            f"Callable '{spec.source}' returned unsupported type {type(result).__name__}; expected Series or DataFrame."
        )
