#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from frequency_utils import (
    BarFrequency,
    add_bar_frequency_argument,
    choose_fullversion_root,
    parse_bar_frequency_or_exit,
)

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

MAX_WINDOW = 10_000
MAX_AST_DEPTH = 20
DEFAULT_WINDOW_NA_THRESH = 0.2

ALLOWED_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
}

ALLOWED_FUNCTIONS = {
    "shift",
    "delay",
    "ts_mean",
    "ts_std",
    "rolling_max",
    "rolling_min",
    "ema",
    "zscore",
    "ret",
    "logret",
    "rank",
    "abs",
    "sign",
    "pow",
    "scale",
    "clip",
    "where",
    "ts_sum",
    "ts_median",
    "ts_prod",
    "ts_skew",
    "ts_kurt",
    "ts_quantile",
    "ts_rank",
    "ts_argmax",
    "ts_argmin",
    "ts_decay_linear",
    "ts_max",
    "ts_min",
    "ts_var",
    "ts_count",
    "ts_corr",
    "ts_cov",
}

WINDOW_FUNCTIONS = {
    "shift",
    "delay",
    "ts_mean",
    "ts_std",
    "rolling_max",
    "rolling_min",
    "ema",
    "zscore",
    "ret",
    "logret",
    "ts_sum",
    "ts_median",
    "ts_prod",
    "ts_skew",
    "ts_kurt",
    "ts_quantile",
    "ts_rank",
    "ts_argmax",
    "ts_argmin",
    "ts_decay_linear",
    "ts_max",
    "ts_min",
    "ts_var",
    "ts_count",
    "ts_corr",
    "ts_cov",
}

FUNCTION_ARG_COUNTS: Dict[str, Tuple[int, int]] = {
    "shift": (2, 2),
    "delay": (2, 2),
    "ts_mean": (2, 2),
    "ts_std": (2, 2),
    "rolling_max": (2, 2),
    "rolling_min": (2, 2),
    "ema": (2, 2),
    "zscore": (2, 2),
    "ret": (2, 2),
    "logret": (2, 2),
    "rank": (1, 1),
    "abs": (1, 1),
    "sign": (1, 1),
    "pow": (2, 2),
    "scale": (2, 2),
    "clip": (3, 3),
    "where": (3, 3),
    "ts_sum": (2, 2),
    "ts_median": (2, 2),
    "ts_prod": (2, 2),
    "ts_skew": (2, 2),
    "ts_kurt": (2, 2),
    "ts_quantile": (3, 3),
    "ts_rank": (2, 3),
    "ts_argmax": (2, 2),
    "ts_argmin": (2, 2),
    "ts_decay_linear": (2, 2),
    "ts_max": (2, 2),
    "ts_min": (2, 2),
    "ts_var": (2, 2),
    "ts_count": (2, 2),
    "ts_corr": (3, 3),
    "ts_cov": (3, 3),
}

WINDOW_ARG_INDEX: Dict[str, int] = {
    "shift": 1,
    "delay": 1,
    "ts_mean": 1,
    "ts_std": 1,
    "rolling_max": 1,
    "rolling_min": 1,
    "ema": 1,
    "zscore": 1,
    "ret": 1,
    "logret": 1,
    "ts_sum": 1,
    "ts_median": 1,
    "ts_prod": 1,
    "ts_skew": 1,
    "ts_kurt": 1,
    "ts_quantile": 1,
    "ts_rank": 1,
    "ts_argmax": 1,
    "ts_argmin": 1,
    "ts_decay_linear": 1,
    "ts_max": 1,
    "ts_min": 1,
    "ts_var": 1,
    "ts_count": 1,
    "ts_corr": 2,
    "ts_cov": 2,
}

__all__ = [
    'MAX_WINDOW',
    'MAX_AST_DEPTH',
    'DEFAULT_WINDOW_NA_THRESH',
    'ALLOWED_FIELDS',
    'ALLOWED_FUNCTIONS',
    'WINDOW_FUNCTIONS',
    'FUNCTION_ARG_COUNTS',
    'WINDOW_ARG_INDEX',
    'SecurityViolation',
    'DSLParseError',
    'IdealMappingResult',
    'ParsedDSL',
    'IdealToDSLMapper',
    'min_periods',
    'make_function_env',
    'normalize_dsl',
    'parse_dsl',
    'evaluate_expression',
    'main',
    'run_main',
]

# ---------------------------------------------------------------------------
# Exceptions and utility classes
# ---------------------------------------------------------------------------


class SecurityViolation(RuntimeError):
    """Raised when attempting to write outside the OUTPUT_DIR."""


class DSLParseError(ValueError):
    """Raised when DSL parsing or validation fails."""


@dataclass
class IdealMappingResult:
    dsl: Optional[str]
    candidates: List[str]
    assumptions: List[str]
    warnings: List[str]


@dataclass
class ParsedDSL:
    factor_name: str
    expression: str
    ast_expr: ast.AST
    required_fields: Set[str]
    used_functions: Set[str]
    windows: List[int]
    complexity: int


class SafePathManager:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root.resolve()
        self.logger: Optional[logging.Logger] = None

    def set_logger(self, logger: logging.Logger) -> None:
        self.logger = logger

    def prepare_path(self, relative_path: Path | str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.output_root / candidate).resolve()
        if not resolved.is_relative_to(self.output_root):
            self._log_violation(resolved)
            raise SecurityViolation(f"Attempted write outside OUTPUT_DIR: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def _log_violation(self, target: Path) -> None:
        message = f"SECURITY_VIOLATION: attempted write to {target}"
        if self.logger:
            self.logger.error(message)
        else:
            print(message, file=sys.stderr)


# ---------------------------------------------------------------------------
# ideal -> DSL mapper
# ---------------------------------------------------------------------------


class IdealToDSLMapper:
    WINDOW_REGEX = re.compile(r"(\d{1,5})")

    def map(self, text: str) -> IdealMappingResult:
        original = text
        normalized = text.strip()
        lower = normalized.lower()
        candidates: List[str] = []
        assumptions: List[str] = []
        warnings: List[str] = []
        best: Optional[str] = None

        window = self._extract_window(lower)

        if any(token in lower for token in ["premium", "溢价"]):
            if any(token in lower for token in ["zscore", "z 分数", "z分数"]):
                w = window or 60
                assumptions.append(f"假设窗口 {w}")
                best = f"PREM_Z_{w} = zscore(premium_close,{w})"
                candidates.append(best)

        if any(token in lower for token in ["log", "对数", "对数收益"]):
            n = window or 1
            assumptions.append(f"假设周期 {n}")
            candidate = f"RET_LOG_{n} = logret(close,{n})"
            candidates.append(candidate)
            if best is None:
                best = candidate

        if any(token in lower for token in ["成交量", "volume", "量能"]):
            w = window or 120
            assumptions.append(f"假设布林窗口 {w}")
            candidate = f"VOL_PCT_{w} = volume / ts_mean(volume,{w})"
            candidates.append(candidate)
            if best is None:
                best = candidate

        if any(token in lower for token in ["布林", "boll", "布林带"]):
            w = window or 100
            assumptions.append(f"假设布林窗口 {w}")
            candidate = (
                f"BB_PCT_{w} = (close - ts_mean(close,{w})) / (2*ts_std(close,{w})) + 0.5"
            )
            candidates.append(candidate)
            if best is None:
                best = candidate

        if not candidates:
            warnings.append("无法 ideal 文本映射成 DSL，请使用 --dsl 手动指定")

        return IdealMappingResult(dsl=best, candidates=candidates, assumptions=assumptions, warnings=warnings)

    def _extract_window(self, text: str) -> Optional[int]:
        match = self.WINDOW_REGEX.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None


# ---------------------------------------------------------------------------
# DSL parsing and validation
# ---------------------------------------------------------------------------


class DSLValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.fields: Set[str] = set()
        self.functions: Set[str] = set()
        self.windows: List[int] = []
        self.max_depth = 0
        self.current_depth = 0

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        if self.max_depth > MAX_AST_DEPTH:
            raise DSLParseError(f"DSL 表达式嵌套深度超过 {MAX_AST_DEPTH}")
        super().visit(node)
        self.current_depth -= 1

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            if node.id in ALLOWED_FUNCTIONS:
                return
            if node.id in ALLOWED_FIELDS:
                self.fields.add(node.id)
                return
            if node.id in {"True", "False"}:
                return
            raise DSLParseError(f"不允许的标识符: {node.id}")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if not isinstance(node.func, ast.Name):
            raise DSLParseError("不支持关键字参数")
        func_name = node.func.id
        if func_name not in ALLOWED_FUNCTIONS:
            raise DSLParseError(f"函数 {func_name} 未在白名单中")
        self.functions.add(func_name)
        if node.keywords:
            raise DSLParseError("不支持关键字参数")
        arg_count = len(node.args)
        if func_name in FUNCTION_ARG_COUNTS:
            min_args, max_args = FUNCTION_ARG_COUNTS[func_name]
            if arg_count < min_args or arg_count > max_args:
                if min_args == max_args:
                    raise DSLParseError(f"函数 {func_name} 需要 {min_args} 个参数，实际收到 {arg_count} 个")
                raise DSLParseError(
                    f"函数 {func_name} 参数个数必须在 [{min_args}, {max_args}] 范围，实际收到 {arg_count} 个"
                )
        for arg in node.args:
            self.visit(arg)
        if func_name in WINDOW_FUNCTIONS:
            window_index = WINDOW_ARG_INDEX[func_name]
            if arg_count <= window_index:
                raise DSLParseError(f"函数 {func_name} 未在白名单中")
            window_arg = node.args[window_index]
            if not isinstance(window_arg, ast.Constant) or not isinstance(window_arg.value, (int, float)):
                raise DSLParseError(f"函数 {func_name} 的窗口参数必须为数值常量")
            window_value = int(window_arg.value)
            if window_value <= 0:
                raise DSLParseError("不支持关键字参数")
            if window_value > MAX_WINDOW:
                raise DSLParseError(f"窗口大小 {window_value} 超过上限 {MAX_WINDOW}")
            self.windows.append(window_value)
        if func_name == "ts_quantile":
            q_arg = node.args[2]
            if not isinstance(q_arg, ast.Constant) or not isinstance(q_arg.value, (int, float)):
                raise DSLParseError("ts_quantile 的分位参数必须是常量")
            q_value = float(q_arg.value)
            if not 0.0 <= q_value <= 1.0:
                raise DSLParseError("ts_quantile 的分位必须位于 [0, 1] 区间")
        if func_name == "ts_rank" and arg_count == 3:
            pct_arg = node.args[2]
            if not isinstance(pct_arg, ast.Constant) or not isinstance(pct_arg.value, (bool, int, float)):
                raise DSLParseError("ts_rank 的 pct 参数必须是布尔或数值常量")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        raise DSLParseError("禁止访问对象属性")

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        raise DSLParseError("禁止使用下标访问")

    def generic_visit(self, node: ast.AST) -> None:
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Num,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Compare,
            ast.Gt,
            ast.GtE,
            ast.Lt,
            ast.LtE,
            ast.Eq,
            ast.NotEq,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.IfExp,
        )
        if not isinstance(node, allowed_nodes):
            # Names and Calls handled separately
            if not isinstance(node, (ast.Name, ast.Call)):
                raise DSLParseError(f"不支持的语法节点: {type(node).__name__}")
        super().generic_visit(node)


def normalize_dsl(dsl: str) -> str:
    return re.sub(r"\s+", "", dsl)


def parse_dsl(source: str) -> ParsedDSL:
    if "=" not in source:
        raise DSLParseError("DSL 格式需满足 <NAME> = <EXPR>")
    name_part, expr_part = source.split("=", 1)
    factor_name = name_part.strip()
    if not factor_name:
        raise DSLParseError("窗口参数必须为正整数")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", factor_name):
        raise DSLParseError("因子名称须以大写字母开头，仅包含大写字母、数字或下划线")

    expression = expr_part.strip()
    if not expression:
        raise DSLParseError("表达式不能为空")

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise DSLParseError(f"DSL 语法错误: {exc.msg}") from exc

    validator = DSLValidator()
    validator.visit(parsed)

    return ParsedDSL(
        factor_name=factor_name,
        expression=expression,
        ast_expr=parsed.body,
        required_fields=set(validator.fields),
        used_functions=set(validator.functions),
        windows=validator.windows,
        complexity=validator.max_depth,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def min_periods(window: int, na_thresh: float) -> int:
    valid_ratio = max(0.0, 1.0 - na_thresh)
    return max(1, int(math.ceil(valid_ratio * window)))


def make_function_env(na_thresh: float) -> Dict[str, Any]:
    def _shift(x: pd.Series, n: int) -> pd.Series:
        return x.shift(int(n))

    def _delay(x: pd.Series, n: int) -> pd.Series:
        return _shift(x, n)

    def _ts_mean(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).mean()

    def _ts_std(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).std(ddof=1)

    def _rolling_max(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).max()

    def _rolling_min(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).min()

    def _ema(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.ewm(span=window, adjust=False, min_periods=mp).mean()

    def _zscore(x: pd.Series, n: int) -> pd.Series:
        mean = _ts_mean(x, n)
        std = _ts_std(x, n)
        return (x - mean) / std

    def _ret(x: pd.Series, n: int) -> pd.Series:
        shifted = _shift(x, n)
        return x / shifted - 1

    def _logret(x: pd.Series, n: int) -> pd.Series:
        shifted = _shift(x, n)
        return np.log(x) - np.log(shifted)

    def _rank(x: pd.Series) -> pd.Series:
        return x.rank(method='average', pct=True)

    def _abs(x: pd.Series) -> pd.Series:
        return x.abs()

    def _sign(x: pd.Series) -> pd.Series:
        return pd.Series(np.sign(x.to_numpy()), index=x.index)

    def _pow(x: pd.Series | float, y: pd.Series | float) -> pd.Series:
        if isinstance(x, pd.Series) and isinstance(y, pd.Series):
            aligned_x, aligned_y = x.align(y, join='outer')
            return aligned_x.pow(aligned_y)
        if isinstance(x, pd.Series):
            return x.pow(y)
        if isinstance(y, pd.Series):
            return y.rpow(x)
        raise DSLParseError('pow 至少需要一个 Series 参数')

    def _scale(x: pd.Series, k: pd.Series | float | int) -> pd.Series:
        denom = x.abs().sum()
        if denom == 0 or pd.isna(denom):
            return pd.Series(0.0, index=x.index)
        scaled = x / denom
        if isinstance(k, pd.Series):
            k_aligned = k.reindex(x.index)
            return (scaled * k_aligned).reindex(x.index)
        return scaled * float(k)

    def _clip(
        x: pd.Series,
        lower: pd.Series | float | int,
        upper: pd.Series | float | int,
    ) -> pd.Series:
        if isinstance(lower, pd.Series):
            lower = lower.reindex(x.index)
        if isinstance(upper, pd.Series):
            upper = upper.reindex(x.index)
        return x.clip(lower=lower, upper=upper)

    def _where(cond: pd.Series, a: pd.Series | float, b: pd.Series | float) -> pd.Series:
        cond_index = cond.index
        if isinstance(a, pd.Series):
            a = a.reindex(cond_index)
        if isinstance(b, pd.Series):
            b = b.reindex(cond_index)
        return pd.Series(np.where(cond.values, a, b), index=cond_index)

    def _ts_sum(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).sum()

    def _ts_median(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).median()

    def _ts_prod(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)

        def _prod(values: np.ndarray) -> float:
            arr = np.asarray(values, dtype=float)
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                return np.nan
            # rolling.apply keeps alignment while ignoring NaN values
            return float(np.prod(valid))

        return x.rolling(window=window, min_periods=mp).apply(_prod, raw=True)

    def _ts_skew(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).skew()

    def _ts_kurt(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).kurt()

    def _ts_quantile(x: pd.Series, n: int, q: float) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).quantile(float(q))

    def _ts_rank(x: pd.Series, n: int, _pct: int | float | bool | None = None) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)

        def _rank_last(values: np.ndarray) -> float:
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return np.nan
            last = arr[-1]
            if np.isnan(last):
                return np.nan
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                return np.nan
            return float((valid <= last).sum() / valid.size)

        return x.rolling(window=window, min_periods=mp).apply(_rank_last, raw=True)

    def _ts_argmax(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)

        def _argmax(values: np.ndarray) -> float:
            arr = np.asarray(values, dtype=float)
            valid_idx = np.flatnonzero(~np.isnan(arr))
            if valid_idx.size == 0:
                return np.nan
            target = valid_idx[np.argmax(arr[valid_idx])]
            return float(arr.size - target - 1)

        return x.rolling(window=window, min_periods=mp).apply(_argmax, raw=True)

    def _ts_argmin(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)

        def _argmin(values: np.ndarray) -> float:
            arr = np.asarray(values, dtype=float)
            valid_idx = np.flatnonzero(~np.isnan(arr))
            if valid_idx.size == 0:
                return np.nan
            target = valid_idx[np.argmin(arr[valid_idx])]
            return float(arr.size - target - 1)

        return x.rolling(window=window, min_periods=mp).apply(_argmin, raw=True)

    def _ts_decay_linear(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        weights = np.arange(1, window + 1, dtype=float)

        def _decay(values: np.ndarray) -> float:
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                return np.nan
            window_weights = weights[-arr.size:]
            valid = ~np.isnan(arr)
            if not valid.any():
                return np.nan
            chosen_weights = window_weights[valid]
            weight_sum = chosen_weights.sum()
            if weight_sum == 0:
                return np.nan
            chosen_values = arr[valid]
            return float(np.dot(chosen_values, chosen_weights) / weight_sum)

        return x.rolling(window=window, min_periods=mp).apply(_decay, raw=True)

    def _ts_max(x: pd.Series, n: int) -> pd.Series:
        return _rolling_max(x, n)

    def _ts_min(x: pd.Series, n: int) -> pd.Series:
        return _rolling_min(x, n)

    def _ts_var(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).var(ddof=1)

    def _ts_count(x: pd.Series, n: int) -> pd.Series:
        window = int(n)
        mp = min_periods(window, na_thresh)
        return x.rolling(window=window, min_periods=mp).count()

    def _ts_corr(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        if not isinstance(y, pd.Series):
            raise DSLParseError('ts_corr 的第二个参数必须是 Series')
        window = int(n)
        mp = min_periods(window, na_thresh)
        y_aligned = y.reindex(x.index)
        return x.rolling(window=window, min_periods=mp).corr(y_aligned)

    def _ts_cov(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
        if not isinstance(y, pd.Series):
            raise DSLParseError('ts_cov 的第二个参数必须是 Series')
        window = int(n)
        mp = min_periods(window, na_thresh)
        y_aligned = y.reindex(x.index)
        return x.rolling(window=window, min_periods=mp).cov(y_aligned)

    return {
        'shift': _shift,
        'delay': _delay,
        'ts_mean': _ts_mean,
        'ts_std': _ts_std,
        'rolling_max': _rolling_max,
        'rolling_min': _rolling_min,
        'ema': _ema,
        'zscore': _zscore,
        'ret': _ret,
        'logret': _logret,
        'rank': _rank,
        'abs': _abs,
        'sign': _sign,
        'pow': _pow,
        'scale': _scale,
        'clip': _clip,
        'where': _where,
        'ts_sum': _ts_sum,
        'ts_median': _ts_median,
        'ts_prod': _ts_prod,
        'ts_skew': _ts_skew,
        'ts_kurt': _ts_kurt,
        'ts_quantile': _ts_quantile,
        'ts_rank': _ts_rank,
        'ts_argmax': _ts_argmax,
        'ts_argmin': _ts_argmin,
        'ts_decay_linear': _ts_decay_linear,
        'ts_max': _ts_max,
        'ts_min': _ts_min,
        'ts_var': _ts_var,
        'ts_count': _ts_count,
        'ts_corr': _ts_corr,
        'ts_cov': _ts_cov,
    }

def evaluate_expression(parsed: ParsedDSL, data: pd.DataFrame, na_thresh: float) -> pd.Series:
    env = make_function_env(na_thresh)
    local_vars: Dict[str, Any] = {field: data[field] for field in parsed.required_fields}
    local_vars.update(env)
    local_vars["np"] = np

    compiled = compile(ast.Expression(parsed.ast_expr), filename="<dsl>", mode="eval")
    result = eval(compiled, {"__builtins__": {}}, local_vars)  # noqa: S307
    if not isinstance(result, pd.Series):
        if isinstance(result, (int, float)):
            return pd.Series(result, index=data.index)
        raise DSLParseError("表达式结果必须为 Series 类型")
    return result.rename(parsed.factor_name)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def expand_symbol_aliases(symbol: str) -> List[str]:
    aliases: List[str] = []

    def add(alias: str) -> None:
        if alias and alias not in aliases:
            aliases.append(alias)

    add(symbol)
    if symbol.endswith("PERP") and len(symbol) > 4:
        add(symbol[:-4])
    stable_suffixes = ["USDT", "BUSD", "USDC", "TUSD", "FDUSD"]
    for current in list(aliases):
        for suffix in stable_suffixes:
            if current.endswith(suffix) and len(current) > len(suffix):
                add(current[: -len(suffix)])
    return aliases


def ensure_datetime(date_str: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(date_str, tz="UTC")
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{date_str}': {exc}") from exc


def date_range_inclusive(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if end < start:
        raise SystemExit("end date must be >= start date")
    days = (end - start).days
    return [start + pd.Timedelta(days=offset) for offset in range(days + 1)]


def find_fullversion_file(directory: Path, symbol: str, day: pd.Timestamp) -> Optional[Path]:
    aliases = expand_symbol_aliases(symbol)
    for alias in aliases:
        pattern = f"{alias}_{day.strftime('%Y_%m_%d')}_*_Fullversion.csv"
        candidates = sorted(directory.glob(pattern))
        if candidates:
            return candidates[0]
    return None


def _extract_start_ms_from_filename(path: Path) -> int:
    for part in reversed(path.stem.split("_")):
        if part.isdigit():
            try:
                return int(part)
            except ValueError:
                continue
    return 0


def _suffix_matches(path: Path, bar_frequency: BarFrequency, allow_suffix_mismatch: bool) -> bool:
    stem = path.stem
    if stem.endswith(f"Fullversion_{bar_frequency.canonical}"):
        return True
    if allow_suffix_mismatch and stem.endswith("Fullversion"):
        return True
    return False


def _collect_fullversion_files(
    directory: Path,
    symbol: str,
    bar_frequency: BarFrequency,
    allow_suffix_mismatch: bool,
) -> List[Path]:
    aliases = expand_symbol_aliases(symbol)
    seen: Dict[Path, int] = {}
    for alias in aliases:
        for item in directory.glob(f"{alias}_*_Fullversion*.csv"):
            if not _suffix_matches(item, bar_frequency, allow_suffix_mismatch):
                continue
            seen.setdefault(item, _extract_start_ms_from_filename(item))
    return sorted(seen.keys(), key=lambda p: seen[p])


def _build_expected_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar_frequency: BarFrequency,
) -> pd.DatetimeIndex:
    if end <= start:
        return pd.DatetimeIndex([], tz="UTC", name="open_time")
    last_inclusive = end - bar_frequency.delta
    if last_inclusive < start:
        return pd.DatetimeIndex([], tz="UTC", name="open_time")
    index = pd.date_range(start=start, end=last_inclusive, freq=bar_frequency.offset, tz="UTC")
    index.name = "open_time"
    return index


def _one_day_span(bar_frequency: BarFrequency) -> pd.Timedelta:
    """Return the duration of a full day expressed in the configured bar frequency."""
    return bar_frequency.to_timedelta(bar_frequency.bars_per_day_int())


def _empty_period(
    symbol: str,
    columns: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar_frequency: BarFrequency,
) -> pd.DataFrame:
    index = _build_expected_index(start, end, bar_frequency)
    df = pd.DataFrame(index=index)
    df.index.name = "open_time"
    df["symbol"] = symbol
    for column in columns:
        df[column] = np.nan
    return df[["symbol"] + list(columns)]


def load_symbol_data(
    fullversion_dir: Path,
    symbol: str,
    required_fields: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    logger: logging.Logger,
    bar_frequency: BarFrequency,
    allow_suffix_mismatch: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not fullversion_dir.exists():
        raise FileNotFoundError(f"Fullversion directory not found: {fullversion_dir}")

    needed_columns = sorted(set(required_fields))
    usecols = ["open_time"] + needed_columns
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)

    files = _collect_fullversion_files(fullversion_dir, symbol, bar_frequency, allow_suffix_mismatch)
    stats: Dict[str, Any] = {"missing_days": [], "files": []}

    if not files:
        logger.warning("No Fullversion files found for %s", symbol)
        empty = _empty_period(symbol, needed_columns, start, end, bar_frequency)
        stats["files"] = []
        stats["expected_rows"] = len(empty)
        stats["available_rows"] = 0
        stats["coverage"] = 0.0
        return empty, stats

    frames: List[pd.DataFrame] = []
    for path in files:
        stats["files"].append(str(path))
        logger.info("Reading %s for %s", path.name, symbol)
        try:
            iterator = pd.read_csv(path, usecols=usecols, chunksize=500_000)
        except ValueError:
            iterator = pd.read_csv(path, chunksize=500_000)
        for chunk in iterator:
            if "open_time" not in chunk.columns:
                continue
            mask = (chunk["open_time"] >= start_ms) & (chunk["open_time"] < end_ms)
            if not mask.any():
                continue
            subset = chunk.loc[mask, [col for col in chunk.columns if col == "open_time" or col in needed_columns]].copy()
            frames.append(subset)

    if not frames:
        logger.warning("No rows for %s between %s and %s", symbol, start, end)
        empty = _empty_period(symbol, needed_columns, start, end, bar_frequency)
        stats["expected_rows"] = len(empty)
        stats["available_rows"] = 0
        stats["coverage"] = 0.0
        return empty, stats

    combined = pd.concat(frames, ignore_index=True)
    combined["open_time"] = pd.to_datetime(combined["open_time"], unit="ms", utc=True, errors="coerce")
    combined = combined.dropna(subset=["open_time"])
    combined = combined.set_index("open_time").sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    expected_index = _build_expected_index(start, end, bar_frequency)
    combined = combined.reindex(expected_index)
    combined.index.name = "open_time"
    combined["symbol"] = symbol
    for column in needed_columns:
        if column not in combined.columns:
            combined[column] = np.nan
    combined = combined[["symbol"] + list(needed_columns)]

    stats["expected_rows"] = len(combined)
    stats["available_rows"] = int(combined[needed_columns[0]].notna().sum()) if needed_columns and len(combined) else 0
    stats["coverage"] = float(stats["available_rows"] / stats["expected_rows"]) if stats["expected_rows"] else 0.0
    return combined, stats




def _empty_day(
    symbol: str, columns: Sequence[str], day: pd.Timestamp, bar_frequency: BarFrequency
) -> pd.DataFrame:
    day_span = _one_day_span(bar_frequency)  # derive full-day window from bar spacing
    index = _build_expected_index(day, day + day_span, bar_frequency)
    df = pd.DataFrame(index=index)
    df.index.name = "open_time"
    df["symbol"] = symbol
    for column in columns:
        df[column] = np.nan
    return df[["symbol"] + list(columns)]


def _reindex_day(
    symbol: str, columns: Sequence[str], df: pd.DataFrame, day: pd.Timestamp, bar_frequency: BarFrequency
) -> pd.DataFrame:
    day_span = _one_day_span(bar_frequency)  # derive full-day window from bar spacing
    index = _build_expected_index(day, day + day_span, bar_frequency)
    if df.empty:
        base = pd.DataFrame(index=index)
    else:
        df = df.copy()
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
        df = df.set_index("open_time").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        base = df.reindex(index)
    base.index.name = "open_time"
    base["symbol"] = symbol
    for column in columns:
        if column not in base.columns:
            base[column] = np.nan
    return base[["symbol"] + list(columns)]


def _empty_range(
    symbol: str,
    columns: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar_frequency: BarFrequency,
) -> pd.DataFrame:
    day_span = _one_day_span(bar_frequency)  # include the entire final day via bar-based offset
    index = _build_expected_index(start, end + day_span, bar_frequency)
    df = pd.DataFrame(index=index)
    df["symbol"] = symbol
    for column in columns:
        df[column] = np.nan
    return df[["symbol"] + list(columns)]


# ---------------------------------------------------------------------------
# Factor computation pipeline
# ---------------------------------------------------------------------------


def compute_cache_key(symbol: str, normalized_dsl: str, start: str, end: str) -> str:
    payload = f"{symbol}|{normalized_dsl}|{start}|{end}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_series_parquet(manager: SafePathManager, path: Path, series: pd.Series) -> Path:
    df = series.to_frame(name=series.name)
    df.index.name = "open_time"
    target = manager.prepare_path(path)
    df.to_parquet(target, engine=_detect_parquet_engine())
    return target


def build_factor_frame_qlib(series: pd.Series, symbol: str, factor_name: str) -> pd.DataFrame:
    renamed = series.rename(factor_name)
    df = renamed.to_frame(name=f"${factor_name}")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    if df.empty:
        empty_index = pd.MultiIndex.from_arrays(
            [
                pd.DatetimeIndex([], tz="UTC"),
                pd.Index([], dtype=object),
            ],
            names=["datetime", "instrument"],
        )
        df.index = empty_index
        return df
    multi = pd.MultiIndex.from_arrays(
        [df.index, pd.Index([symbol] * len(df), name="instrument")],
        names=["datetime", "instrument"],
    )
    df.index = multi
    return df


def write_factor_symbol_qlib(
    manager: SafePathManager,
    features_dir: Path,
    factor_name: str,
    symbol: str,
    frame: pd.DataFrame,
) -> Path:
    relative_path = features_dir / factor_name / f"{symbol}.parquet"
    target = manager.prepare_path(relative_path)
    frame.to_parquet(target, engine=_detect_parquet_engine())
    return relative_path


def write_factor_long_qlib(
    manager: SafePathManager,
    features_dir: Path,
    factor_name: str,
    frames: List[pd.DataFrame],
) -> Optional[Path]:
    if not frames:
        return None
    combined = pd.concat(frames).sort_index()
    long_df = combined.reset_index()
    value_column = f"${factor_name}"
    if value_column in long_df.columns:
        long_df = long_df.rename(columns={value_column: factor_name})
    relative_path = features_dir / f"{factor_name}_long.parquet"
    target = manager.prepare_path(relative_path)
    long_df.to_parquet(target, engine=_detect_parquet_engine(), index=False)
    return relative_path


def build_step3_qlib_summary_qlib(
    run_id: str,
    factor_name: str,
    features_dir: Path,
    symbol_paths: Dict[str, str],
    long_path: Optional[Path],
    summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    valid_rates = [row.get("valid_rate") for row in summaries if isinstance(row.get("valid_rate"), (int, float))]
    avg_valid = float(sum(valid_rates) / len(valid_rates)) if valid_rates else None
    payload = {
        "run_id": run_id,
        "factor": factor_name,
        "outputs": {
            "symbol_dir": str(features_dir / factor_name),
            "long": str(long_path) if long_path else None,
        },
        "symbol_files": symbol_paths,
        "statistics": {
            "symbol_count": len(symbol_paths),
            "valid_rate_avg": avg_valid,
        },
        "summaries": summaries,
    }
    return payload


def _detect_parquet_engine() -> str:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return "fastparquet"
        except ImportError as exc:  # noqa: F841
            raise RuntimeError("Neither pyarrow nor fastparquet is installed")


def write_summary(
    manager: SafePathManager, summary_dir: Path, run_id: str, rows: List[Dict[str, Any]]
) -> None:
    df = pd.DataFrame(rows)
    safe_path = summary_dir / f"summary_{run_id}.csv"
    safe_write_csv(manager, df, safe_path)


def safe_write_csv(manager: SafePathManager, df: pd.DataFrame, relative_path: Path | str) -> Path:
    target = manager.prepare_path(relative_path)
    df.to_csv(target, index=False)
    return target


def safe_write_json(manager: SafePathManager, relative_path: Path | str, payload: Dict[str, Any]) -> Path:
    target = manager.prepare_path(relative_path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def safe_write_text(manager: SafePathManager, relative_path: Path | str, content: str) -> Path:
    target = manager.prepare_path(relative_path)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return target


# ---------------------------------------------------------------------------
# CLI and main execution
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factor Engine V1")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dsl")
    parser.add_argument("--ideal")
    parser.add_argument("--symbols", required=True, help="Comma separated symbols")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--window-na-thresh", type=float, default=DEFAULT_WINDOW_NA_THRESH)
    parser.add_argument("--dry-run", type=str, default="false")
    parser.add_argument("--strict-readonly", type=str, default="true")
    parser.add_argument("--enable-cache", type=str, default="false")
    parser.add_argument("--run-id")
    add_bar_frequency_argument(parser)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def setup_logger(manager: SafePathManager, run_id: str, bar_frequency: BarFrequency) -> logging.Logger:
    log_path = manager.prepare_path(Path("logs") / bar_frequency.canonical / f"{run_id}.log")
    logger = logging.getLogger("step3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def gather_environment_snapshot(extra_packages: Iterable[str]) -> Dict[str, Any]:
    import platform

    from importlib import metadata as importlib_metadata

    packages: Dict[str, Optional[str]] = {}
    for name in extra_packages:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "packages": packages}


def main() -> None:
    args = parse_arguments()
    bar_frequency = parse_bar_frequency_or_exit(args.bar_freq)
    args.bar_freq = bar_frequency.canonical
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or time.strftime("%Y%m%d%H%M%S")

    manager = SafePathManager(output_dir)
    logger = setup_logger(manager, run_id, bar_frequency)
    manager.set_logger(logger)

    fullversion_dir, using_legacy = choose_fullversion_root(data_root, bar_frequency, logger=logger)

    freq_token = bar_frequency.canonical
    artifacts_dir = Path("artifacts") / freq_token
    factors_root = Path("factors") / freq_token
    cache_root = Path("factors_cache") / freq_token
    qlib_root = Path("qlib") / freq_token
    qlib_features_dir = qlib_root / "features"
    snapshot_root = Path("config_snapshots") / freq_token

    strict_readonly = parse_bool(args.strict_readonly)
    enable_cache = parse_bool(args.enable_cache)
    dry_run = parse_bool(args.dry_run)
    na_thresh = float(args.window_na_thresh)

    logger.info("Factor engine start run_id=%s bar_freq=%s", run_id, bar_frequency.canonical)
    logger.info("Parameters: symbols=%s start=%s end=%s", args.symbols, args.start, args.end)

    if args.dsl is None and args.ideal is None:
        raise SystemExit("请在 --dsl 与 --ideal 之间二选一")

    mapper = IdealToDSLMapper()
    mapping_result = None
    dsl_source: Optional[str] = args.dsl
    if dsl_source is None and args.ideal is not None:
        mapping_result = mapper.map(args.ideal)
        if mapping_result.dsl is None:
            raise SystemExit("ideal 内容无法映射为 DSL，请使用 --dsl 指定")
        dsl_source = mapping_result.dsl
    normalized_dsl = normalize_dsl(dsl_source)
    parsed = parse_dsl(dsl_source)

    parse_artifact = {
        "run_id": run_id,
        "ideal": args.ideal,
        "dsl": dsl_source,
        "normalized_dsl": normalized_dsl,
        "assumptions": mapping_result.assumptions if mapping_result else [],
        "candidates": mapping_result.candidates if mapping_result else [],
        "warnings": mapping_result.warnings if mapping_result else [],
        "required_fields": sorted(parsed.required_fields),
        "used_functions": sorted(parsed.used_functions),
        "windows": parsed.windows,
        "complexity": parsed.complexity,
        "bar_freq": bar_frequency.canonical,
    }
    safe_write_json(manager, artifacts_dir / f"parse_result_{run_id}.json", parse_artifact)

    if dry_run:
        logger.info("Dry run enabled; skipping computation")
        return

    start_ts = ensure_datetime(args.start)
    end_ts = ensure_datetime(args.end)

    symbols = [sym.strip() for sym in args.symbols.split(",") if sym.strip()]
    summaries: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    factor_name = parsed.factor_name
    qlib_factor_frames: List[pd.DataFrame] = []
    qlib_symbol_rel_paths: Dict[str, str] = {}

    for symbol in symbols:
        logger.info("Processing symbol %s", symbol)
        try:
            data, load_stats = load_symbol_data(
                fullversion_dir,
                symbol,
                parsed.required_fields,
                start_ts,
                end_ts,
                logger,
                bar_frequency,
                using_legacy,
            )
        except FileNotFoundError as exc:
            logger.warning("%s", exc)
            skipped.append({"symbol": symbol, "reason": str(exc)})
            continue

        if not load_stats.get("files"):
            reason = "缺少 Fullversion 数据"
            logger.warning("Skipping %s: %s", symbol, reason)
            skipped.append({"symbol": symbol, "reason": reason})
            continue

        factor_dir = factors_root / factor_name
        cache_dir = cache_root / factor_name

        cache_series: Optional[pd.Series] = None
        cache_hit = False
        cache_rel_path: Optional[Path] = None
        if enable_cache:
            cache_key = compute_cache_key(symbol, normalized_dsl, args.start, args.end)
            cache_rel_path = cache_dir / f"{cache_key}.parquet"
            cache_abs_path = manager.prepare_path(cache_rel_path)
            if cache_abs_path.exists():
                logger.info("Cache hit for %s", symbol)
                cache_df = pd.read_parquet(cache_abs_path)
                if parsed.factor_name in cache_df.columns:
                    cache_series = cache_df[parsed.factor_name]
                    cache_series.index = pd.to_datetime(cache_series.index, utc=True)
                    cache_hit = True

        start_time = time.perf_counter()
        if cache_series is not None:
            result = cache_series.rename(parsed.factor_name)
        else:
            result = evaluate_expression(parsed, data, na_thresh)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        valid_rate = float(result.notna().mean()) if len(result) else 0.0
        logger.info("Symbol %s valid_rate=%.4f rows=%d time=%.2fms", symbol, valid_rate, len(result), elapsed_ms)

        factor_path = factor_dir / f"{symbol}.parquet"
        write_series_parquet(manager, factor_path, result)

        factor_frame_qlib = build_factor_frame_qlib(result, symbol, factor_name)
        relative_feature_path = write_factor_symbol_qlib(manager, qlib_features_dir, factor_name, symbol, factor_frame_qlib)
        qlib_symbol_rel_paths[symbol] = str(relative_feature_path)
        if not factor_frame_qlib.empty:
            qlib_factor_frames.append(factor_frame_qlib)

        if enable_cache and not cache_hit and cache_rel_path is not None:
            write_series_parquet(manager, cache_rel_path, result)

        summaries.append(
            {
                "symbol": symbol,
                "rows": len(result),
                "valid_rate": valid_rate,
                "needed_columns": ",".join(sorted(parsed.required_fields)),
                "runtime_ms": elapsed_ms,
                "first_ts": result.index.min().isoformat() if len(result.index) else None,
                "last_ts": result.index.max().isoformat() if len(result.index) else None,
                "cache_hit": cache_hit,
            }
        )

    qlib_long_path = write_factor_long_qlib(manager, qlib_features_dir, factor_name, qlib_factor_frames)

    if summaries:
        write_summary(manager, factors_root / factor_name, run_id, summaries)
    if skipped:
        skipped_df = pd.DataFrame(skipped)
        safe_write_csv(manager, skipped_df, Path("artifacts") / f"skipped_symbols_{run_id}.csv")

    snapshot_dir = snapshot_root / run_id
    env_snapshot = gather_environment_snapshot(["pandas", "numpy"])
    config_payload = {
        "run_id": run_id,
        "parameters": vars(args),
        "parsed": parse_artifact,
        "summaries": summaries,
        "skipped": skipped,
        "environment": env_snapshot,
        "qlib_outputs": {
            "symbol_files": qlib_symbol_rel_paths,
            "long_file": str(qlib_long_path) if qlib_long_path else None,
        },
    }
    config_payload["parameters"]["bar_freq"] = bar_frequency.canonical
    safe_write_json(manager, snapshot_dir / "step3_args.json", config_payload)

    qlib_summary_payload = build_step3_qlib_summary_qlib(
        run_id,
        factor_name,
        qlib_features_dir,
        qlib_symbol_rel_paths,
        qlib_long_path,
        summaries,
    )
    safe_write_json(manager, snapshot_dir / "step3_qlib_summary.json", qlib_summary_payload)
    logger.info("Factor engine completed run_id=%s", run_id)


def run_main() -> int:
    try:
        main()
        return 0
    except SecurityViolation as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except DSLParseError as exc:
        print(f"DSL ERROR: {exc}", file=sys.stderr)
        return 3
    except SystemExit as exc:
        raise exc
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("step3").exception("Unhandled error: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run_main())
