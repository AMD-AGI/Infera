###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Load predictors: forecast the next interval from the observed history.

Scaling on the interval that just ended means always arriving one interval
late, which for a rising ramp is exactly when the SLA is missed. These
predictors turn the observed series (request count, ISL, OSL) into a forecast
for the interval about to start.

Two are built in and need nothing beyond numpy:

  * ``constant`` -- next interval looks like the last one. Right choice when
    the adjustment interval is long relative to how fast traffic moves.
  * ``ewma`` -- exponentially weighted moving average, which rides through
    single-interval noise but lags a sustained ramp.

The interface matches Dynamo's planner predictors (``add_data_point`` /
``predict_next``) so a heavier forecaster -- ARIMA, Prophet -- can be dropped
into :data:`PREDICTORS` later without touching the planner loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    """A single scalar time series with a one-step-ahead forecast."""

    def __init__(self, *, window_size: int = 10) -> None:
        self._window_size = max(1, window_size)
        self._history: list[float] = []

    def add_data_point(self, value: float) -> None:
        """Append an observation.

        Leading zeros are dropped: before traffic arrives the metrics window is
        empty, and letting those zeros into the history would hold the forecast
        down through the first real interval of load.
        """
        if not self._history and value <= 0:
            return
        self._history.append(float(value))
        if len(self._history) > self._window_size:
            del self._history[: -self._window_size]

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def get_last_value(self) -> float:
        """Most recent observation, or 0.0 if nothing has been observed."""
        return self._history[-1] if self._history else 0.0

    @abstractmethod
    def predict_next(self) -> float:
        """Forecast for the interval about to start."""


class ConstantPredictor(BasePredictor):
    """Assume the next interval repeats the last one."""

    def predict_next(self) -> float:
        return self.get_last_value()


class EwmaPredictor(BasePredictor):
    """Exponentially weighted moving average over the retained history.

    ``alpha`` is the weight on the newest observation; higher tracks change
    faster and smooths less. At 1.0 this degenerates to
    :class:`ConstantPredictor`.
    """

    def __init__(self, *, window_size: int = 10, alpha: float = 0.5) -> None:
        super().__init__(window_size=window_size)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"ewma alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha

    def predict_next(self) -> float:
        if not self._history:
            return 0.0
        estimate = self._history[0]
        for value in self._history[1:]:
            estimate = self._alpha * value + (1.0 - self._alpha) * estimate
        return estimate


PREDICTORS: dict[str, type[BasePredictor]] = {
    "constant": ConstantPredictor,
    "ewma": EwmaPredictor,
}


def build_predictor(name: str, *, window_size: int = 10) -> BasePredictor:
    """Instantiate the named predictor, or raise with the valid names listed."""
    try:
        cls = PREDICTORS[name]
    except KeyError:
        raise ValueError(
            f"unknown load predictor {name!r}; choose one of {sorted(PREDICTORS)}"
        ) from None
    return cls(window_size=window_size)
