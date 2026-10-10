"""Entidade de domínio para Violação de EPI confirmada."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from src.core.domain.ppe_type import ViolationClass
from src.core.domain.vehicle_metadata import VehicleMetadata


@dataclass
class Violation:
    """Representa uma violação de segurança confirmada pelas regras de conformidade."""
    violation_type: str
    track_id: Optional[int]
    timestamp: float
    consecutive_frames: int
    confidence: float
    bbox: Tuple[int, int, int, int]
    metadata: Optional[VehicleMetadata] = None

    @property
    def description(self) -> str:
        """Descrição legível da violação em português."""
        try:
            return ViolationClass(self.violation_type).label_pt
        except ValueError:
            return f"Sem {self.violation_type.replace('no_', '').capitalize()} 🔴"

    @property
    def unique_key(self) -> str:
        """Chave de identificação única para controle de frequência / rate limiting."""
        prefix = ""
        if self.metadata and self.metadata.plate:
            prefix = f"{self.metadata.plate}_"
        elif self.metadata and self.metadata.video_name:
            prefix = f"{self.metadata.video_name}_"

        track_str = str(self.track_id) if self.track_id is not None else "unknown"
        return f"{prefix}{self.violation_type}_track_{track_str}"
