"""Controle de frequência e cooldown para alertas."""

import logging
import threading
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Evita disparo repetitivo de alertas dentro de uma janela de tempo (cooldown)."""

    def __init__(self, cooldown_seconds: int = 60):
        self.cooldown = max(0, cooldown_seconds)
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()  # chamado por várias threads (câmeras, API, Desktop)

    def can_send(self, key: str) -> bool:
        """Verifica se a chave atingiu o tempo mínimo desde o último disparo."""
        if self.cooldown == 0:
            return True

        now = time.monotonic()  # imune a ajustes do relógio do sistema
        with self._lock:
            last = self._last_sent.get(key)
            if last is None or now - last >= self.cooldown:
                self._last_sent[key] = now
                if len(self._last_sent) > 10_000:  # evita crescimento sem limite
                    cutoff = now - self.cooldown
                    self._last_sent = {k: t for k, t in self._last_sent.items() if t >= cutoff}
                return True

        remaining = self.cooldown - (now - last)
        logger.debug(f"⏱️ Cooldown ativo para '{key}'. Restam {remaining:.0f}s.")
        return False

    def reset(self) -> None:
        """Limpa o histórico de envios."""
        with self._lock:
            self._last_sent.clear()
