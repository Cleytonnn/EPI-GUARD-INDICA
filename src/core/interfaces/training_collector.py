"""Interface (porta) para coleta de amostras de treino durante a análise (Aprendizado Ativo)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation


class ITrainingSampleCollector(ABC):
    """Seleciona e guarda frames relevantes para revisão humana e retreino."""

    @abstractmethod
    def start_session(self, source_name: str) -> None:
        """Inicia uma nova sessão (novo vídeo ou nova câmera), zerando contadores locais."""
        raise NotImplementedError

    @abstractmethod
    def consider(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        violations: Optional[List[Violation]] = None,
        metadata: Optional[VehicleMetadata] = None,
    ) -> Optional[str]:
        """Avalia o frame e, se for relevante, salva como amostra pendente.

        Returns:
            ID da amostra salva, ou None se o frame foi descartado.
        """
        raise NotImplementedError

