"""Typed progress events emitted by the engine.

UI-agnostic on purpose: the engine emits these through an optional
callback and never imports a rendering library. Consumers (PlainReporter,
Dashboard) live in featherquant.ui.
"""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Phase:
    """A setup/teardown step worth narrating (metadata read, planning,
    budget sizing, resume verification). Fired before streaming begins."""
    label: str


@dataclass(frozen=True)
class JobStart:
    """Fired once, after planning, before the first tensor streams."""
    total_tensors: int
    total_in_bytes: int
    total_out_bytes: int
    done_out_bytes: int   # committed bytes when resuming (bar starts here)
    max_ram: int
    fmt: str
    dst: str
    resumed_at: int       # index of first tensor this run writes (0 = fresh)


@dataclass(frozen=True)
class TensorStart:
    """Fired before each tensor begins streaming."""
    index: int
    name: str
    src_type: str
    dst_type: str
    out_bytes: int


@dataclass(frozen=True)
class ChunkDone:
    """Fired after every streamed chunk (quantize or copy)."""
    in_bytes: int
    out_bytes: int
    rss: int


@dataclass(frozen=True)
class TensorDone:
    """Fired after a tensor is committed to the manifest."""
    index: int


@dataclass(frozen=True)
class JobDone:
    """Fired last, with the final stats dict."""
    stats: dict[str, Any]


Event = Phase | JobStart | TensorStart | ChunkDone | TensorDone | JobDone
ProgressFn = Callable[[Event], None]
