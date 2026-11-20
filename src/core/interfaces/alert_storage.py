"""Contrato da porta de Armazenamento de Imagens e Evidências."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.core.domain.violation import Violation


@runtime_checkable
class IAlertStorage(Protocol):
    """Interface abstrata para persistência de evidências visuais de infrações."""

    def save_image(self, frame: np.ndarray, violation: Violation) -> str | None:
        """Salva a imagem da violação anotada e retorna o caminho salvo."""
        ...
