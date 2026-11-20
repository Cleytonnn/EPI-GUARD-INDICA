"""Suavização temporal do estado de EPIs por pessoa rastreada.

Resolve a instabilidade quadro-a-quadro (EPI "piscando" entre presente/ausente)
combinando:

* **Média móvel exponencial (EMA)** da evidência de presença do EPI;
* **Histerese** (limiares diferentes para entrar/sair de um estado);
* **Estado "unknown"** enquanto não há evidência suficiente — nesse caso nenhuma
  violação é emitida (evita falsos alertas de EPI que está sendo usado);
* **Suavização das caixas** (EMA nas coordenadas) para um desenho estável.

Classe pura de domínio: sem dependências de IA/visão computacional.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum

Box = tuple[int, int, int, int]


class PPEState(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


@dataclass
class _Track:
    ema: float
    state: PPEState = PPEState.UNKNOWN
    observations: int = 1
    last_seen: int = 0
    box: tuple[float, float, float, float] | None = None


class PPEStateSmoother:
    """Mantém o estado estável (presente/ausente/desconhecido) de cada (pessoa, EPI)."""

    def __init__(
        self,
        alpha: float = 0.35,
        present_on: float = 0.55,
        absent_on: float = 0.33,
        min_observations: int = 3,
        box_alpha: float = 0.4,
        max_age: int = 90,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha deve estar em (0, 1]")
        if absent_on >= present_on:
            raise ValueError("absent_on deve ser menor que present_on (histerese)")
        self.alpha = alpha
        self.present_on = present_on
        self.absent_on = absent_on
        self.min_observations = max(1, min_observations)
        self.box_alpha = box_alpha
        self.max_age = max_age
        self._tracks: dict[Hashable, _Track] = {}
        self._frame = 0

    def begin_frame(self) -> None:
        """Avança o relógio interno e descarta chaves que sumiram há muito tempo."""
        self._frame += 1
        stale = [k for k, t in self._tracks.items() if self._frame - t.last_seen > self.max_age]
        for key in stale:
            del self._tracks[key]

    def update(self, key: Hashable, observation: float, box: Box | None = None) -> tuple[PPEState, float, Box | None]:
        """Atualiza a evidência (0 = ausente, 1 = presente) e devolve (estado, score, caixa suavizada)."""
        observation = min(1.0, max(0.0, float(observation)))
        track = self._tracks.get(key)
        if track is None:
            track = _Track(ema=observation, last_seen=self._frame)
            self._tracks[key] = track
        else:
            track.ema = self.alpha * observation + (1.0 - self.alpha) * track.ema
            track.observations += 1
            track.last_seen = self._frame

        if box is not None:
            if track.box is None:
                track.box = tuple(float(v) for v in box)
            else:
                a = self.box_alpha
                track.box = tuple(a * n + (1.0 - a) * o for n, o in zip(box, track.box, strict=True))

        if track.observations >= self.min_observations:
            if track.ema >= self.present_on:
                track.state = PPEState.PRESENT
            elif track.ema <= self.absent_on:
                track.state = PPEState.ABSENT
            # Entre os limiares: mantém o estado anterior (histerese)

        smooth_box = tuple(int(round(v)) for v in track.box) if track.box is not None else None
        return track.state, track.ema, smooth_box

    def reset(self) -> None:
        self._tracks.clear()
        self._frame = 0
