"""Serviço de conformidade e validação temporal de infrações de EPI."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation

logger = logging.getLogger(__name__)

# Ordem de avaliação e geometria das caixas inferidas: (fator de confiança, topo, base) relativos à pessoa
_INFERENCE_ORDER = ("helmet", "gloves", "vest", "uniform_rf")
_INFERRED_ZONES: dict[str, tuple[float, float, float]] = {
    "helmet": (0.85, 0.0, 0.45),
    "gloves": (0.80, 0.40, 0.90),
    "vest": (0.85, 0.15, 0.70),
    "uniform_rf": (0.85, 0.15, 0.85),
}


class ComplianceService:
    """Validador de conformidade temporal de EPIs.

    Aplica janela deslizante e exigência de persistência por N frames
    consecutivos para filtrar oclusões temporárias e falsos positivos.
    """

    def __init__(
        self,
        required_ppe: list[str] | None = None,
        violation_classes: list[str] | None = None,
        min_violation_frames: int = 5,
        infer_missing_ppe: bool = True,
        only_company_workers: bool = True,
    ):
        self.required_ppe = set(required_ppe or ["helmet", "gloves", "goggles", "vest", "boots"])
        self.violation_classes = set(
            violation_classes or ["no_helmet", "no_gloves", "no_boots", "no_goggle", "no_vest", "no_uniform_rf"]
        )
        self.min_violation_frames = max(1, min_violation_frames)
        self.infer_missing_ppe = infer_missing_ppe
        self.only_company_workers = bool(only_company_workers)

        self._violation_counter: dict[tuple, int] = defaultdict(int)
        self._confidence_sum: dict[tuple, float] = defaultdict(float)
        self._last_bbox: dict[tuple, tuple] = {}
        self._current_frame_keys: set[tuple] = set()
        self._confirmed_violations: set[tuple] = set()

    def _is_ppe_on_person(self, person_bbox: tuple, ppe_bbox: tuple, ppe_type: str = "helmet") -> bool:
        """Verifica se o EPI está na região anatômica correspondente da pessoa."""
        px1, py1, px2, py2 = person_bbox
        height = max(1, py2 - py1)
        hx1, hy1, hx2, hy2 = ppe_bbox
        hcx = (hx1 + hx2) / 2
        hcy = (hy1 + hy2) / 2

        # Tolerância horizontal (mãos e membros podem estender-se levemente além do tronco)
        margin_x = max(35, int(0.25 * (px2 - px1)))
        if not (px1 - margin_x <= hcx <= px2 + margin_x):
            return False

        if ppe_type in {"helmet", "no_helmet"}:
            head_y2 = py1 + int(height * 0.50)
            return py1 - 30 <= hcy <= head_y2 + 25
        elif ppe_type in {"vest", "no_vest", "uniform_rf", "no_uniform_rf"}:
            torso_y1 = py1 + int(height * 0.15)
            torso_y2 = py1 + int(height * 0.75)
            return torso_y1 - 25 <= hcy <= torso_y2 + 25
        elif ppe_type in {"gloves", "no_gloves"}:
            hands_y1 = py1 + int(height * 0.30)
            return hands_y1 - 30 <= hcy <= py2 + 30
        elif ppe_type in {"boots", "no_boots"}:
            feet_y1 = py1 + int(height * 0.65)
            return feet_y1 - 30 <= hcy <= py2 + 40
        else:
            return py1 - 30 <= hcy <= py2 + 30

    def check(
        self,
        detections: list[Detection],
        metadata: VehicleMetadata | None = None,
    ) -> list[Violation]:
        """Avalia as detecções do frame atual e retorna violações confirmadas."""
        if not detections:
            # Quadro vazio: todas as infrações em contagem "sumiram". Antes o conjunto do quadro
            # anterior não era limpo, então nada era zerado e a contagem continuava depois.
            self._current_frame_keys.clear()
            self._decay_counters()
            return []

        active_detections = list(detections)
        is_cabin = bool(metadata and getattr(metadata, "is_cabin", False))

        # Regra de Conformidade: Pessoa detectada sem EPI obrigatório
        if self.infer_missing_ppe:
            # Audita apenas trabalhadores ('Person'), ignorando civis/pedestres ('civil')
            persons = [d for d in detections if d.class_name == "Person"]
            for ppe in _INFERENCE_ORDER:
                if ppe not in self.required_ppe:
                    continue
                present = [d for d in detections if d.class_name == ppe]
                direct_no = [d for d in detections if d.class_name == f"no_{ppe}"]
                factor, top, bottom = _INFERRED_ZONES[ppe]
                for person in persons:
                    if any(self._is_ppe_on_person(person.bbox, d.bbox, ppe) for d in present):
                        continue
                    if any(self._is_ppe_on_person(person.bbox, d.bbox, f"no_{ppe}") for d in direct_no):
                        continue
                    px1, py1, px2, py2 = person.bbox
                    h = py2 - py1
                    y1 = py1 + int(h * top)
                    y2 = py1 + max(int(h * bottom), 20) if ppe == "helmet" else min(py2, py1 + int(h * bottom))
                    active_detections.append(
                        Detection(
                            class_name=f"no_{ppe}",
                            confidence=round(person.confidence * factor, 3),
                            bbox=(px1, y1, px2, y2),
                            track_id=person.track_id,
                        )
                    )

        confirmed: list[Violation] = []
        self._current_frame_keys.clear()

        # Identifica civis para descartar qualquer alerta em transeuntes na rua.
        # Só IDs reais: sem tracking todos têm track_id=None e um único civil no quadro
        # silenciava as infrações de TODOS os trabalhadores.
        civil_tracks = {d.track_id for d in active_detections if d.class_name == "civil" and d.track_id is not None}

        for det in active_detections:
            if det.class_name not in self.violation_classes:
                continue

            # Se for civil na rua, descarta violação
            if self.only_company_workers and not is_cabin:
                if det.track_id is not None and det.track_id in civil_tracks:
                    continue

            key = (det.track_id, det.class_name)
            # Conta no máximo 1 vez por quadro: sem tracking, N pessoas com a mesma infração
            # compartilham a chave e o contador subia N vezes por quadro (confirmação precoce)
            if key in self._current_frame_keys:
                continue
            self._current_frame_keys.add(key)
            self._violation_counter[key] += 1
            self._confidence_sum[key] += det.confidence
            self._last_bbox[key] = det.bbox

            if self._violation_counter[key] >= self.min_violation_frames and key not in self._confirmed_violations:
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
