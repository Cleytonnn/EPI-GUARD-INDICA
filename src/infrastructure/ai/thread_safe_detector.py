"""Decorator que serializa o acesso ao detector entre threads.

A API (FastAPI roda rotas síncronas em um pool de threads), o monitor de câmeras
HTTP (uma thread por câmera) e o app Desktop (análise de arquivo + ao vivo) podem
chamar o mesmo detector ao mesmo tempo. Os modelos Ultralytics com ``track(persist=True)``
e o ``PPEStateSmoother`` guardam estado interno e não são thread-safe.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from src.core.domain.detection import Detection


class ThreadSafeDetector:
    """Envolve qualquer ``IDetector`` com um ``RLock``; demais atributos são repassados."""

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_lock", threading.RLock())

    @property
    def inner(self) -> Any:
        return self._inner

    def detect(self, frame: np.ndarray, skip_frames: int = 1) -> list[Detection]:
        with self._lock:
            return self._inner.detect(frame, skip_frames)

    def reset_tracking(self) -> None:
        with self._lock:
            self._inner.reset_tracking()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Ex.: detector.is_cabin = True / detector.confidence = 0.3 → aplicados no detector real
        setattr(self._inner, name, value)
