"""Entidades de domínio para Detecção de Objetos e EPIs."""

from __future__ import annotations

import builtins
from dataclasses import dataclass

from src.core.domain.ppe_type import PPEClass, ViolationClass


@dataclass(frozen=True)
class BoundingBox:
    """Coordenadas retangulares da detecção (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def tuple(self) -> builtins.tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Detection:
    """Entidade que representa uma detecção efetuada pelo modelo de IA."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int | None = None

    @property
    def is_violation(self) -> bool:
        """Indica se a detecção representa ausência de EPI."""
        return self.class_name.startswith("no_")

    @property
    def ppe_type(self) -> str:
        """Retorna o tipo de EPI associado à detecção."""
        if self.is_violation:
            try:
                return ViolationClass(self.class_name).related_ppe
            except ValueError:
                return self.class_name.replace("no_", "")
        return self.class_name

    @property
    def label_pt(self) -> str:
        """Rótulo legível em português."""
        try:
            if self.is_violation:
                return ViolationClass(self.class_name).label_pt
            return PPEClass(self.class_name).label_pt
        except ValueError:
            return self.class_name

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(*self.bbox)
