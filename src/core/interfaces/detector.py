"""Contrato da porta de Detecção de Objetos e EPIs."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable
import numpy as np
from src.core.domain.detection import Detection


@runtime_checkable
class IDetector(Protocol):
    """Interface abstrata para detectores de visão computacional."""

    def detect(self, frame: np.ndarray, skip_frames: int = 1) -> List[Detection]:
        """Processa um frame e retorna lista de detecções."""
        ...

    def reset_tracking(self) -> None:
        """Reseta o estado de tracking de objetos (ByteTrack)."""
        ...

    @property
    def confidence(self) -> float:
        """Limiar de confiança mínima configurado."""
        ...

    @property
    def class_names(self) -> dict:
        """Mapeamento de IDs para nomes de classes do modelo."""
        ...
