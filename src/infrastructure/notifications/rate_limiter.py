"""Controle de frequência e cooldown para alertas."""

import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Evita disparo repetitivo de alertas dentro de uma janela de tempo (cooldown)."""

    def __init__(self, cooldown_seconds: int = 60):
        self.cooldown = max(0, cooldown_seconds)
        self._last_sent: dict[str, float] = {}

    def can_send(self, key: str) -> bool:
        """Verifica se a chave atingiu o tempo mínimo desde o último disparo."""
        if self.cooldown == 0:
            return True

        now = time.time()
        last = self._last_sent.get(key, 0.0)
        if now - last >= self.cooldown:
            self._last_sent[key] = now
            return True

        remaining = self.cooldown - (now - last)
        logger.debug(f"⏱️ Cooldown ativo para '{key}'. Restam {remaining:.0f}s.")
        return False

    def reset(self) -> None:
        """Limpa o histórico de envios."""
        self._last_sent.clear()
