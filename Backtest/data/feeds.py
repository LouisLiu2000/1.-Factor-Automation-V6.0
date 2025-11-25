"""Backtrader data feed extensions that expose factor columns as indicator lines."""

from __future__ import annotations

from typing import Iterable, Optional

import backtrader as bt


BASE_PARAMS = (
    ("datetime", None),
    ("open", -1),
    ("high", -1),
    ("low", -1),
    ("close", -1),
    ("volume", -1),
    ("openinterest", -1),
)


class FactorDataFeed(bt.feeds.PandasData):
    """Expose arbitrary factor columns as additional lines on a PandasData feed."""

    lines = tuple()
    params = BASE_PARAMS
    datafields = bt.feeds.PandasData.datafields

    @classmethod
    def with_factors(cls, factor_columns: Iterable[str], *, timeframe: Optional[str] = None):
        """Return a subclass publishing the provided factor columns as lines."""

        columns = tuple(str(col) for col in factor_columns if str(col))
        suffix = "_".join(columns) if columns else "base"
        name = f"{cls.__name__}_{suffix}"
        base_params = cls.params._gettuple() if hasattr(cls.params, "_gettuple") else tuple(cls.params)
        params = tuple(list(base_params) + [(col, -1) for col in columns])
        base_datafields = tuple(cls.datafields) if not isinstance(cls.datafields, list) else tuple(cls.datafields)
        datafields = tuple(list(base_datafields) + [col for col in columns if col not in base_datafields])

        attrs = {
            "lines": columns,
            "params": params,
            "datafields": datafields,
            "factor_columns": columns,
        }
        if timeframe is not None:
            attrs["timeframe_hint"] = timeframe
        return type(name, (cls,), attrs)
