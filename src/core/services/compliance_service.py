"""Serviço de conformidade e validação temporal de infrações de EPI."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation

logger = logging.getLogger(__name__)


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
        self._current_frame_keys.clear()
        if not detections:
            # Zera contadores de ausências que sumiram
            self._decay_counters()
            return []

        active_detections = list(detections)
        is_cabin = bool(metadata and getattr(metadata, "is_cabin", False))

        # Regra de Conformidade: Pessoa detectada sem EPI obrigatório
        if self.infer_missing_ppe:
            # Audita apenas trabalhadores ('Person'), ignorando civis/pedestres ('civil')
            persons = [d for d in detections if d.class_name == "Person"]

            # 1. Capacete (Helmet)
            if "helmet" in self.required_ppe:
                helmets = [d for d in detections if d.class_name == "helmet"]
                direct_no_helmets = [d for d in detections if d.class_name == "no_helmet"]
                for person in persons:
                    has_helmet = any(self._is_ppe_on_person(person.bbox, h.bbox, "helmet") for h in helmets)
                    has_direct_no = any(
                        self._is_ppe_on_person(person.bbox, nh.bbox, "no_helmet") for nh in direct_no_helmets
                    )

                    if not has_helmet and not has_direct_no:
                        px1, py1, px2, py2 = person.bbox
                        head_y2 = py1 + max(int((py2 - py1) * 0.45), 20)
                        active_detections.append(
                            Detection(
                                class_name="no_helmet",
                                confidence=round(person.confidence * 0.85, 3),
                                bbox=(px1, py1, px2, head_y2),
                                track_id=person.track_id,
                            )
                        )

            # 2. Luvas (Gloves)
            if "gloves" in self.required_ppe:
                gloves = [d for d in detections if d.class_name == "gloves"]
                direct_no_gloves = [d for d in detections if d.class_name == "no_gloves"]
                for person in persons:
                    has_gloves = any(self._is_ppe_on_person(person.bbox, g.bbox, "gloves") for g in gloves)
                    has_direct_no = any(
                        self._is_ppe_on_person(person.bbox, ng.bbox, "no_gloves") for ng in direct_no_gloves
                    )

                    if not has_gloves and not has_direct_no:
                        px1, py1, px2, py2 = person.bbox
                        h = py2 - py1
                        hands_y1 = py1 + int(h * 0.40)
                        hands_y2 = min(py2, py1 + int(h * 0.90))
                        active_detections.append(
                            Detection(
                                class_name="no_gloves",
                                confidence=round(person.confidence * 0.80, 3),
                                bbox=(px1, hands_y1, px2, hands_y2),
                                track_id=person.track_id,
                            )
                        )

            # 3. Colete (Vest)
            if "vest" in self.required_ppe:
                vests = [d for d in detections if d.class_name == "vest"]
                direct_no_vests = [d for d in detections if d.class_name == "no_vest"]
                for person in persons:
                    has_vest = any(self._is_ppe_on_person(person.bbox, v.bbox, "vest") for v in vests)
                    has_direct_no = any(
                        self._is_ppe_on_person(person.bbox, nv.bbox, "no_vest") for nv in direct_no_vests
                    )

                    if not has_vest and not has_direct_no:
                        px1, py1, px2, py2 = person.bbox
                        h = py2 - py1
                        torso_y1 = py1 + int(h * 0.15)
                        torso_y2 = py1 + int(h * 0.70)
                        active_detections.append(
                            Detection(
                                class_name="no_vest",
                                confidence=round(person.confidence * 0.85, 3),
                                bbox=(px1, torso_y1, px2, torso_y2),
                                track_id=person.track_id,
                            )
                        )

            # 4. Uniforme RF (Uniforme Risco Fogo NR-10)
            if "uniform_rf" in self.required_ppe:
                uniforms = [d for d in detections if d.class_name == "uniform_rf"]
                direct_no_uniforms = [d for d in detections if d.class_name == "no_uniform_rf"]
                for person in persons:
                    has_uniform = any(self._is_ppe_on_person(person.bbox, u.bbox, "uniform_rf") for u in uniforms)
                    has_direct_no = any(
                        self._is_ppe_on_person(person.bbox, nu.bbox, "no_uniform_rf") for nu in direct_no_uniforms
                    )

                    if not has_uniform and not has_direct_no:
                        px1, py1, px2, py2 = person.bbox
                        h = py2 - py1
                        torso_y1 = py1 + int(h * 0.15)
                        torso_y2 = min(py2, py1 + int(h * 0.85))
                        active_detections.append(
                            Detection(
                                class_name="no_uniform_rf",
                                confidence=round(person.confidence * 0.85, 3),
                                bbox=(px1, torso_y1, px2, torso_y2),
                                track_id=person.track_id,
                            )
                        )

        confirmed: list[Violation] = []

        # Identifica civis com tracking válido para descartar alertas em transeuntes na rua
        # (civis sem track_id / -1 nunca devem silenciar trabalhadores sem track_id)
        civil_tracks = {
            d.track_id
            for d in active_detections
            if d.class_name == "civil" and d.track_id is not None and d.track_id >= 0
        }

        # Deduplica por (track_id, class_name) dentro do mesmo frame para evitar contagem dupla
        # (ex.: duas mãos sem luva ou múltiplas detecções sem tracking no mesmo frame)
        frame_candidates: dict[tuple, Detection] = {}
        for det in active_detections:
            if det.class_name not in self.violation_classes:
                continue

            # Se for civil na rua, descarta violação
            if self.only_company_workers and not is_cabin:
                if det.track_id is not None and det.track_id in civil_tracks:
                    continue

            key = (det.track_id, det.class_name)
            prev = frame_candidates.get(key)
            if prev is None or det.confidence > prev.confidence:
                frame_candidates[key] = det

        for key, det in frame_candidates.items():
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
