"""Contrato da porta de Leitura e Decodificação de Vídeos."""

from __future__ import annotations

from typing import Optional, Protocol, Tuple, runtime_checkable
import numpy as np


@runtime_checkable
class IVideoReader(Protocol):
    """Interface abstrata para decodificadores de fluxo de vídeo."""

    def is_opened(self) -> bool:
        """Indica se o arquivo ou stream está aberto com sucesso."""
        ...

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lê o próximo frame como array NumPy BGR."""
        ...

    def release(self) -> None:
        """Libera os recursos e fecha o arquivo ou stream."""
        ...

    @property
    def fps(self) -> float:
        """Taxa de quadros por segundo."""
        ...

    @property
    def total_frames(self) -> int:
        """Total de frames estimados (ou 0 se indeterminado)."""
        ...

    @property
    def width(self) -> int:
        """Largura do vídeo em pixels."""
        ...

    @property
    def height(self) -> int:
        """Altura do vídeo em pixels."""
        ...

    @property
    def codec_name(self) -> str:
        """Nome do codec detectado (ex: hevc, h264)."""
        ...
