"""Serviço de conformidade e validação temporal de infrações de EPI."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set
from src.core.domain.detection import Detection
from src.core.domain.violation import Violation
from src.core.domain.vehicle_metadata import VehicleMetadata

logger = logging.getLogger(__name__)


class ComplianceService:
    """Validador de conformidade temporal de EPIs.
    
    Aplica janela deslizante e exigência de persistência por N frames
    consecutivos para filtrar oclusões temporárias e falsos positivos.
    """

    def __init__(
        self,
        required_ppe: Optional[List[str]] = None,
        violation_classes: Optional[List[str]] = None,
        min_violation_frames: int = 5,
    ):
        self.required_ppe = set(required_ppe or ["helmet", "gloves", "goggles", "vest", "boots"])
        self.violation_classes = set(violation_classes or [
            "no_helmet", "no_gloves", "no_boots", "no_goggle", "no_vest"
        ])
        self.min_violation_frames = max(1, min_violation_frames)

        self._violation_counter: Dict[tuple, int] = defaultdict(int)
        self._confidence_sum: Dict[tuple, float] = defaultdict(float)
        self._last_bbox: Dict[tuple, tuple] = {}
        self._current_frame_keys: Set[tuple] = set()
        self._confirmed_violations: Set[tuple] = set()

    def check(
        self,
        detections: List[Detection],
        metadata: Optional[VehicleMetadata] = None,
    ) -> List[Violation]:
        """Avalia as detecções do frame atual e retorna violações confirmadas."""
        if not detections:
            # Reduz contadores de ausências que sumiram
            self._decay_counters()
            return []

        confirmed: List[Violation] = []
        self._current_frame_keys.clear()

        for det in detections:
            if det.class_name not in self.violation_classes:
                continue

            key = (det.track_id, det.class_name)
            self._current_frame_keys.add(key)
            self._violation_counter[key] += 1
            self._confidence_sum[key] += det.confidence
            self._last_bbox[key] = det.bbox

            if (
                self._violation_counter[key] >= self.min_violation_frames
                and key not in self._confirmed_violations
            ):
                count = self._violation_counter[key]
                avg_conf = self._confidence_sum[key] / count

                violation = Violation(
                    violation_type=det.class_name,
                    track_id=det.track_id,
                    timestamp=time.time(),
                    consecutive_frames=count,
                    confidence=round(avg_conf, 3),
                    bbox=det.bbox,
                    metadata=metadata,
                )
                confirmed.append(violation)
                self._confirmed_violations.add(key)
                logger.warning(
                    f"🚨 VIOLAÇÃO CONFIRMADA: {violation.description} "
                    f"(ID: {det.track_id}, Frames: {count}, Conf: {avg_conf:.1%})"
                )

        # Trata ausências
        self._decay_counters()
        return confirmed

    def _decay_counters(self) -> None:
        """Reseta ou decresce contadores de violações que desapareceram."""
        disappeared = set(self._violation_counter.keys()) - self._current_frame_keys
        for key in disappeared:
            if key in self._confirmed_violations:
                self._confirmed_violations.remove(key)
            self._violation_counter.pop(key, None)
            self._confidence_sum.pop(key, None)
            self._last_bbox.pop(key, None)

    def reset(self) -> None:
        """Reseta todo o estado interno."""
        self._violation_counter.clear()
        self._confidence_sum.clear()
        self._last_bbox.clear()
        self._current_frame_keys.clear()
        self._confirmed_violations.clear()
