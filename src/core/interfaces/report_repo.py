"""Contrato para o Repositório de Relatórios de Infrações."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.domain.violation import Violation


@runtime_checkable
class IViolationReportRepository(Protocol):
    """Interface para persistência de relatórios estruturados de infrações (CSV, DB)."""

    def log_violation(self, violation: Violation, image_path: str | None = None) -> None:
        """Registra uma infração confirmada no relatório."""
        ...
