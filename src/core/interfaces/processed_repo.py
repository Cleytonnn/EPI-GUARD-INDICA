"""Contrato para o Repositório de Vídeos Processados (Idempotência)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.core.domain.vehicle_metadata import VehicleMetadata


@runtime_checkable
class IProcessedVideosRepository(Protocol):
    """Interface para consulta e registro de arquivos já analisados."""

    def is_processed(self, video_path: Path | str) -> bool:
        """Verifica se o vídeo já foi processado e permanece inalterado."""
        ...

    def mark_processed(
        self,
        video_path: Path | str,
        metadata: VehicleMetadata,
        violations_count: int,
        duration_seconds: float,
    ) -> None:
        """Registra a conclusão do processamento de um vídeo."""
        ...
