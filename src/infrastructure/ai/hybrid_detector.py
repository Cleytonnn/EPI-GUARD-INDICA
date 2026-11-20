"""Detector híbrido: YOLO-Pose (pessoa/anatomia) + YOLOv8 treinado em EPIs + suavização temporal.

Fluxo por frame:
1. ``PoseEPIDetector`` localiza as pessoas (com tracking) e as zonas anatômicas
   (cabeça, tronco, mãos) com a análise heurística de cor.
2. O modelo YOLOv8 treinado (metodologia do TCC) é executado **em um recorte
   enquadrado de cada pessoa** (padding + upscale), o que melhora muito a detecção
   de itens pequenos (luvas/capacete) em câmeras de CFTV.
3. As evidências do modelo e da heurística são fundidas por item; o resultado passa
   por EMA + histerese (``PPEStateSmoother``). Só se emite ``no_<epi>`` quando a
   ausência é estável — EPI visível pelo modelo **nunca** gera infração.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.core.domain.detection import Detection
from src.core.interfaces.detector import IDetector
from src.core.services.ppe_state_smoother import PPEState, PPEStateSmoother

logger = logging.getLogger(__name__)

Box = tuple[int, int, int, int]

POSITIVE = {"helmet", "gloves", "vest"}
NEGATIVE = {"no_helmet", "no_gloves"}


def expand_box(box: Box, ratio: float, min_px: int, frame_w: int, frame_h: int) -> Box:
    """Expande uma caixa por ``ratio`` da largura/altura (mínimo ``min_px``), limitada ao frame."""
    x1, y1, x2, y2 = box
    dx = max(min_px, int((x2 - x1) * ratio))
    dy = max(min_px, int((y2 - y1) * ratio))
    return max(0, x1 - dx), max(0, y1 - dy), min(frame_w - 1, x2 + dx), min(frame_h - 1, y2 + dy)


def center_in_box(inner: Box, outer: Box) -> bool:
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def default_zones(person: Box) -> dict[str, Box]:
    """Zonas aproximadas a partir da caixa da pessoa (usadas quando não há keypoints)."""
    x1, y1, x2, y2 = person
    _w, h = x2 - x1, y2 - y1
    return {
        "head": (x1, y1, x2, y1 + int(h * 0.28)),
        "torso": (x1, y1 + int(h * 0.15), x2, y1 + int(h * 0.62)),
    }


def fuse_evidence(pos: float, neg: float, heuristic_present: bool | None, has_negative_class: bool) -> float:
    """Converte evidências (modelo + heurística) em uma observação de presença em [0, 1].

    - Modelo viu o EPI/ausência: domina (80%) e a heurística ajusta (20%).
    - Modelo sem evidência: a heurística sozinha só consegue declarar ausência
      de forma fraca; presença é aceita com peso moderado.
    """
    h = 0.5 if heuristic_present is None else (1.0 if heuristic_present else 0.0)
    if pos > 0.0 or neg > 0.0:
        ratio = pos / (pos + neg)
        strength = min(1.0, max(pos, neg) / 0.5)
        model_obs = 0.5 + (ratio - 0.5) * strength
        return 0.8 * model_obs + 0.2 * h
    if heuristic_present is None:
        return 0.5
    if heuristic_present:
        return 0.80
    # Ausência apenas por heurística de cor: observação fraca
    return 0.30 if has_negative_class else 0.25


class _PersonGroup:
    """Agrupa a pessoa e as detecções heurísticas associadas (zonas e vereditos)."""

    def __init__(self, person: Detection, is_civil: bool = False):
        self.person = person
        self.is_civil = is_civil
        self.items: list[Detection] = []

    def first(self, *names: str) -> Detection | None:
        for d in self.items:
            if d.class_name in names:
                return d
        return None

    def all(self, *names: str) -> list[Detection]:
        return [d for d in self.items if d.class_name in names]


class HybridEPIDetector(IDetector):
    """Combina Pose + modelo YOLOv8 de EPIs + suavização temporal."""

    def __init__(
        self,
        pose_detector: IDetector,
        ppe_model_path: str | Path = "models/epi_best.pt",
        ppe_confidence: float = 0.25,
        ppe_iou: float = 0.5,
        ppe_imgsz: int = 640,
        crop_padding: float = 0.18,
        smoother: PPEStateSmoother | None = None,
        ppe_model=None,
    ):
        self.pose = pose_detector
        self.ppe_confidence = float(ppe_confidence)
        self.ppe_iou = float(ppe_iou)
        self.ppe_imgsz = int(ppe_imgsz)
        self.crop_padding = float(crop_padding)
        self.smoother = smoother or PPEStateSmoother()
        self._worker_grace: dict[int, int] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if ppe_model is not None:
            self.ppe_model = ppe_model
        else:
            from ultralytics import YOLO

            logger.info("🦺 Carregando modelo de EPIs em [%s]: %s", self.device.upper(), ppe_model_path)
            self.ppe_model = YOLO(str(ppe_model_path))
        self._ppe_names: dict[int, str] = dict(getattr(self.ppe_model, "names", {}))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _group(base: list[Detection]) -> list[_PersonGroup]:
        """Reagrupa a lista sequencial do PoseEPIDetector (Person/civil seguido de seus itens)."""
        groups: list[_PersonGroup] = []
        current: _PersonGroup | None = None
        for det in base:
            if det.class_name in ("Person", "civil"):
                current = _PersonGroup(det, is_civil=det.class_name == "civil")
                groups.append(current)
            elif current is not None:
                current.items.append(det)
        return groups

    def _run_ppe_model(self, frame: np.ndarray, groups: list[_PersonGroup]) -> list[list[Detection]]:
        """Roda o modelo de EPIs nos recortes enquadrados de cada pessoa (batch)."""
        h, w = frame.shape[:2]
        crops, origins = [], []
        for g in groups:
            x1, y1, x2, y2 = g.person.bbox
            padx = int((x2 - x1) * self.crop_padding)
            pady = int((y2 - y1) * self.crop_padding)
            cx1, cy1 = max(0, x1 - padx), max(0, y1 - pady)
            cx2, cy2 = min(w, x2 + padx), min(h, y2 + pady)
            crop = frame[cy1:cy2, cx1:cx2]
            crops.append(crop if crop.size else np.zeros((32, 32, 3), dtype=np.uint8))
            origins.append((cx1, cy1))
        if not crops:
            return []

        results = self.ppe_model.predict(
            source=crops,
            conf=self.ppe_confidence,
            iou=self.ppe_iou,
            imgsz=self.ppe_imgsz,
            device=self.device,
            verbose=False,
        )
        per_person: list[list[Detection]] = []
        for res, (ox, oy) in zip(results, origins, strict=True):
            dets: list[Detection] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for b, c, k in zip(xyxy, confs, clss, strict=True):
                    name = self._ppe_names.get(int(k), f"class_{k}")
                    dets.append(
                        Detection(
                            class_name=name,
                            confidence=float(c),
                            bbox=(int(b[0]) + ox, int(b[1]) + oy, int(b[2]) + ox, int(b[3]) + oy),
                        )
                    )
            per_person.append(dets)
        return per_person

    @staticmethod
    def _evidence(
        model_dets: list[Detection], zone: Box, pos_names: set, neg_names: set, frame_w: int, frame_h: int
    ) -> tuple[float, float, Box | None]:
        """Maior confiança positiva/negativa do modelo cujo centro cai na zona expandida."""
        area = expand_box(zone, 0.6, 20, frame_w, frame_h)
        pos = neg = 0.0
        pos_box: Box | None = None
        for d in model_dets:
            if not center_in_box(d.bbox, area):
                continue
            if d.class_name in pos_names and d.confidence > pos:
                pos, pos_box = d.confidence, d.bbox
            elif d.class_name in neg_names and d.confidence > neg:
                neg = d.confidence
        return pos, neg, pos_box

    # ------------------------------------------------------------------- detect
    def detect(self, frame: np.ndarray, skip_frames: int = 1) -> list[Detection]:
        base = self.pose.detect(frame, skip_frames)
        self.smoother.begin_frame()
        if frame is None or frame.size == 0 or not base:
            return base

        frame_h, frame_w = frame.shape[:2]
        groups = self._group(base)
        if not groups:
            return base

        try:
            model_by_person = self._run_ppe_model(frame, groups)
        except Exception as exc:  # modelo falhou: mantém o resultado heurístico
            logger.error("Erro no modelo de EPIs, usando apenas heurística: %s", exc, exc_info=True)
            return base

        output: list[Detection] = []
        for idx, (group, model_dets) in enumerate(zip(groups, model_by_person, strict=True)):
            person = group.person
            tid = person.track_id if person.track_id not in (None, -1) else -(idx + 1)

            # Suaviza o enquadramento da própria pessoa para evitar saltos quadro a quadro
            _, _, smooth_person_box = self.smoother.update((tid, "person"), 1.0, person.bbox)
            if smooth_person_box is not None:
                person = Detection(person.class_name, person.confidence, smooth_person_box, person.track_id)

            # Pedestre: só mantém worker por curta janela de graça (evita lock permanente se encostar em cone)
            if group.is_civil:
                strong = [
                    d
                    for d in model_dets
                    if (d.class_name == "helmet" and d.confidence >= 0.58)
                    or (d.class_name in {"vest", "uniform_rf", "coverall"} and d.confidence >= 0.60)
                ]
                grace = self._worker_grace.get(tid, 0) if tid >= 0 else 0
                if not strong and grace <= 0:
                    output.append(person)
                    continue
                if tid >= 0:
                    if strong:
                        self._worker_grace[tid] = min(6, grace + 2)
                    else:
                        self._worker_grace[tid] = grace - 1
                zones = default_zones(person.bbox)
                group.items = []
                person = Detection("Person", person.confidence, person.bbox, person.track_id)
            else:
                zones = {}
                if tid >= 0:
                    self._worker_grace[tid] = min(6, self._worker_grace.get(tid, 0) + 2)

            output.append(person)
            self._emit_items(output, group, model_dets, zones, tid, person, frame_w, frame_h)
        return output

    def _emit_items(
        self,
        output: list[Detection],
        group: _PersonGroup,
        model_dets: list[Detection],
        zones: dict[str, Box],
        tid: int,
        person: Detection,
        frame_w: int,
        frame_h: int,
    ) -> None:
        pid = person.track_id

        def emit(pos_name: str, neg_name: str | None, key: tuple, obs: float, zone: Box) -> PPEState:
            state, score, box = self.smoother.update(key, obs, zone)
            box = box or zone
            if state is PPEState.PRESENT:
                output.append(Detection(pos_name, round(float(score), 3), box, pid))
            elif state is PPEState.ABSENT and neg_name:
                output.append(Detection(neg_name, round(1.0 - float(score), 3), box, pid))
            return state

        # --- Capacete
        h_det = group.first("helmet", "no_helmet")
        head_zone = h_det.bbox if h_det else zones.get("head")
        if head_zone:
            pos, neg, mbox = self._evidence(model_dets, head_zone, {"helmet"}, {"no_helmet"}, frame_w, frame_h)
            heur = None if h_det is None else h_det.class_name == "helmet"
            # Evita que 'no_helmet' fraco do modelo externo sobrescreva capacete confirmado pela Pose (ex.: verde/distante)
            effective_neg = (
                0.0
                if (heur is True and h_det is not None and h_det.confidence >= 0.26 and pos == 0.0 and neg < 0.70)
                else neg
            )
            obs = fuse_evidence(pos, effective_neg, heur, has_negative_class=True)
            emit("helmet", "no_helmet", (tid, "helmet"), obs, mbox if (mbox and pos >= effective_neg) else head_zone)

        # --- Uniforme RF / Colete (Uniforme NR-10 confirmado não é sobrescrito por 'no_vest' de datasets externos)
        u_det = group.first("uniform_rf", "no_uniform_rf")
        torso_zone = u_det.bbox if u_det else zones.get("torso")
        if torso_zone:
            vpos, vneg, vbox = self._evidence(
                model_dets,
                torso_zone,
                {"vest", "uniform_rf", "coverall"},
                {"no_vest", "no_uniform_rf", "no_coverall"},
                frame_w,
                frame_h,
            )
            heur_uniform = None if u_det is None else u_det.class_name == "uniform_rf"
            vneg_for_rf = 0.0 if heur_uniform is True else vneg
            uobs = fuse_evidence(
                vpos if vpos >= 0.45 else 0.0,
                vneg_for_rf,
                heur_uniform,
                has_negative_class=bool(vneg_for_rf > 0),
            )
            u_state = emit("uniform_rf", "no_uniform_rf", (tid, "uniform_rf"), uobs, torso_zone)

            # Evita rótulo duplicado/contraditório de colete quando Uniforme RF já está presente ou ausente no mesmo tronco
            if u_state is PPEState.UNKNOWN and vpos >= 0.45:
                vobs = fuse_evidence(vpos, 0.0, None, has_negative_class=False)
                emit("vest", None, (tid, "vest"), vobs, vbox or torso_zone)

        # --- Luvas (uma por mão visível; fallback no modelo de EPIs se punhos da Pose estiverem incertos)
        cx_person = (person.bbox[0] + person.bbox[2]) / 2.0
        pose_hands = group.all("gloves", "no_gloves")
        if pose_hands:
            for g_det in pose_hands:
                side = "L" if (g_det.bbox[0] + g_det.bbox[2]) / 2.0 < cx_person else "R"
                pos, neg, mbox = self._evidence(model_dets, g_det.bbox, {"gloves"}, {"no_gloves"}, frame_w, frame_h)
                obs = fuse_evidence(pos, neg, g_det.class_name == "gloves", has_negative_class=True)
                emit("gloves", "no_gloves", (tid, f"gloves_{side}"), obs, mbox if (mbox and pos >= neg) else g_det.bbox)
        else:
            for d in model_dets:
                if d.class_name == "gloves" and d.confidence >= 0.45:
                    side = "L" if (d.bbox[0] + d.bbox[2]) / 2.0 < cx_person else "R"
                    emit(
                        "gloves",
                        "no_gloves",
                        (tid, f"gloves_{side}"),
                        fuse_evidence(d.confidence, 0.0, None, True),
                        d.bbox,
                    )
                elif d.class_name == "no_gloves" and d.confidence >= 0.55:
                    side = "L" if (d.bbox[0] + d.bbox[2]) / 2.0 < cx_person else "R"
                    emit(
                        "gloves",
                        "no_gloves",
                        (tid, f"gloves_{side}"),
                        fuse_evidence(0.0, d.confidence, None, True),
                        d.bbox,
                    )

    # ---------------------------------------------------------- IDetector proxy
    def reset_tracking(self) -> None:
        self.pose.reset_tracking()
        self.smoother.reset()
        self._worker_grace.clear()

    @property
    def confidence(self) -> float:
        return self.pose.confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.pose.confidence = value

    @property
    def class_names(self) -> dict:
        return self.pose.class_names
