"""Contrato da porta de Notificação de Alertas."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.domain.violation import Violation


@runtime_checkable
class INotifier(Protocol):
    """Interface abstrata para envio de notificações (Email, Webhook, SMS, etc.)."""

    def notify(self, violation: Violation, image_path: str | None = None) -> bool:
        """Envia notificação sobre uma violação confirmada."""
        ...
