"""Automatic evaluation architecture.

Each Evaluator produces Metrics for one generated audio. Metrics are always automatic *estimates*; the UI labels them
as such and never presents them as human quality. Evaluators that need models not installed report themselves as
unavailable with a reason instead of returning invented numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field


class Metric(BaseModel):
    id: str
    label: str
    value: float
    display: str
    better: str = Field(description="'higher' o 'lower': qué dirección es mejor")
    description: str


class EvaluatorInfo(BaseModel):
    id: str
    label: str
    description: str
    available: bool
    reason: str | None = None


@dataclass
class EvaluationInput:
    audio: np.ndarray
    sample_rate: int
    target_text: str
    language: str | None = None


class EvaluationOutput(BaseModel):
    metrics: list[Metric]
    details: dict = {}


class Evaluator(Protocol):
    id: str
    label: str
    description: str

    def info(self) -> EvaluatorInfo: ...

    def evaluate(self, data: EvaluationInput) -> EvaluationOutput: ...
