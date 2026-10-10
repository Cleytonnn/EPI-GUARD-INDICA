"""Notificador composto com suporte a rate limiting."""

from __future__ import annotations

import logging
from typing import List, Optional
from src.core.domain.violation import Violation
from src.core.interfaces.notifier import INotifier
from src.infrastructure.notifications.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class CompositeNotifier:
    """Orquestrador de múltiplos canais de notificação com rate limiting."""

    def __init__(self, notifiers: Optional[List[INotifier]] = None, rate_limiter: Optional[RateLimiter] = None):
        self.notifiers = notifiers or []
        self.rate_limiter = rate_limiter or RateLimiter(cooldown_seconds=60)

    def add_notifier(self, notifier: INotifier) -> None:
        self.notifiers.append(notifier)

    def notify(self, violation: Violation, image_path: Optional[str] = None) -> bool:
        """Dispara notificações para todos os canais se o rate limit permitir."""
        alert_key = violation.unique_key

        if not self.rate_limiter.can_send(alert_key):
            return False

        logger.info(f"🚨 Disparando alerta: {violation.description} (chave: {alert_key})")
        sent_any = False
        for notifier in self.notifiers:
            try:
                if notifier.notify(violation, image_path):
                    sent_any = True
            except Exception as e:
                logger.error(f"Erro no notificador {type(notifier).__name__}: {e}")

        return sent_any
