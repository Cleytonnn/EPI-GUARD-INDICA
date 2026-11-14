"""Detector anatômico de EPIs baseado em YOLO-Pose.

Inspirado nas soluções de visão computacional anatômica (ex: EPI-Detect):
1. Usa yolov8n-pose para mapear 17 keypoints corporais (ombros, cabeça, pulsos, quadris).
2. Capacete: ancorado no crânio e ombros, com máscara elíptica, exclusão de pele/cabelo
   e teste de superfície lisa (desvio padrão de brilho) e cores industriais.
3. Uniforme RF (NR-10): analisa o tronco entre ombros e bacia buscando o bloco
   laranja fluorescente na pala/ombros e a faixa refletiva cinza/prata horizontal.
4. Luvas: analisa a região das mãos ancorada nos pulsos (keypoints 9 e 10),
   verificando cobertura de material de proteção vs pele humana descoberta.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.core.domain.detection import Detection
from src.core.interfaces.detector import IDetector

logger = logging.getLogger(__name__)

# Índices de keypoints corporais COCO-17
NOSE = 0
EYE_L, EYE_R = 1, 2
EAR_L, EAR_R = 3, 4
SHOULDER_L, SHOULDER_R = 5, 6
ELBOW_L, ELBOW_R = 7, 8
WRIST_L, WRIST_R = 9, 10
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14
ANKLE_L, ANKLE_R = 15, 16


class PoseEPIDetector(IDetector):
    """Implementa IDetector utilizando YOLO-Pose e análise anatômica de EPIs."""

    def __init__(
        self,
        model_path: str | Path = "yolov8n-pose.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        imgsz_near: int = 768,
        track: bool = True,
        helmet_threshold: float = 0.22,
        uniform_rf_threshold: float = 0.12,
        gloves_skin_threshold: float = 0.35,
        only_company_workers: bool = True,
        is_cabin: bool = False,
    ):
        self.model_path = str(model_path)
        self._confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.imgsz_near = int(imgsz_near)
        self.track = bool(track)

        self.helmet_threshold = helmet_threshold
        self.uniform_rf_threshold = uniform_rf_threshold
        self.gloves_skin_threshold = gloves_skin_threshold
        self.only_company_workers = bool(only_company_workers)
        self.is_cabin = bool(is_cabin)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"🦾 Carregando YOLO-Pose em [{self.device.upper()}]: {self.model_path}")
        self.model = YOLO(self.model_path)
        logger.info("🦾 Modelo YOLO-Pose inicializado com sucesso.")

    def _get_keypoint(
        self, kps: np.ndarray, index: int, min_conf: float = 0.35
    ) -> Optional[Tuple[int, int]]:
        """Retorna coordenadas (x, y) de um keypoint se confiável, senão None."""
        if kps is None or index >= len(kps):
            return None
        x, y, conf = kps[index]
        if conf < min_conf or (x <= 0 and y <= 0):
            return None
        return int(x), int(y)

    def _analyze_helmet(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        kps: np.ndarray,
    ) -> Tuple[bool, float, Tuple[int, int, int, int]]:
        """Analisa a região da cabeça para detectar capacete vs cabeça desprotegida."""
        x1, y1, x2, y2 = box
        frame_h, frame_w = frame.shape[:2]

        sl = self._get_keypoint(kps, SHOULDER_L)
        sr = self._get_keypoint(kps, SHOULDER_R)

        # Ancoragem vertical: crânio (topo da caixa) até os ombros
        top = max(0, y1)
        if sl and sr:
            cx = int((sl[0] + sr[0]) / 2)
            span = max(12, int(min(sl[1], sr[1]) - top))
            shoulder_w = abs(sr[0] - sl[0])
            half_w = max(int(shoulder_w * 0.45), int(span * 0.55), 10)
        elif sl or sr:
            shoulder = sl or sr
            cx = shoulder[0]
            span = max(12, int(shoulder[1] - top))
            half_w = max(int(span * 0.55), 10)
        else:
            cx = int((x1 + x2) / 2)
            span = max(12, int((y2 - y1) * 0.25))
            half_w = max(int((x2 - x1) * 0.40), 10)

        hx1 = max(0, cx - half_w)
        hx2 = min(frame_w - 1, cx + half_w)
        # Foca na calota craniana (75% superiores) para nao diluir com pescoco/gola em vistas traseiras
        crown_span = max(14, int(span * 0.78))
        hy1 = top
        hy2 = min(frame_h - 1, top + crown_span)

        if hx2 <= hx1 or hy2 <= hy1:
            return False, 0.0, (x1, y1, x2, y1 + 30)

        head_roi = frame[hy1:hy2, hx1:hx2]
        if head_roi.size == 0:
            return False, 0.0, (hx1, hy1, hx2, hy2)

        roi_h, roi_w = head_roi.shape[:2]

        # Máscara elíptica ajustada à calota do crânio
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        center = (roi_w // 2, roi_h // 2)
        axes = (max(1, roi_w // 2 - 2), max(1, roi_h // 2 - 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        total_pixels = cv2.countNonZero(mask)
        if total_pixels == 0:
            return False, 0.0, (hx1, hy1, hx2, hy2)

        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)

        # 1. Filtro de pele (Skin exclusion)
        lower_skin = np.array([0, 35, 55], dtype=np.uint8)
        upper_skin = np.array([25, 180, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin) & mask

        # 2. Filtro de cabelo escuro / sombra neutra (exclui capacete colorido na sombra)
        dark_hair_mask = (hsv[:, :, 2] < 38) & (hsv[:, :, 1] < 65) & mask

        # 3. Superfície lisa de plástico (desvio padrão de brilho baixo na calota)
        masked_gray = gray[mask > 0]
        std_dev = np.std(masked_gray) if len(masked_gray) > 0 else 999.0

        # 4. Cores industriais de capacete (Branco brilhante, Amarelo, Laranja, Verde, Azul, Vermelho)
        white_mask = (hsv[:, :, 1] < 38) & (hsv[:, :, 2] > 205) & mask

        yellow_orange_mask = (
            cv2.inRange(hsv, np.array([4, 55, 65]), np.array([38, 255, 255])) & mask
        )

        # Verde industrial (ex.: capacete verde de supervisao/eletricista, inclusive em sombra)
        green_mask = (
            cv2.inRange(hsv, np.array([34, 35, 25]), np.array([92, 255, 255])) & mask
        )

        blue_mask = (
            cv2.inRange(hsv, np.array([90, 55, 55]), np.array([132, 255, 255])) & mask
        )

        red1 = cv2.inRange(hsv, np.array([0, 65, 60]), np.array([8, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([168, 65, 60]), np.array([180, 255, 255]))
        red_mask = (red1 | red2) & mask

        all_helmet_colors = white_mask | yellow_orange_mask | green_mask | blue_mask | red_mask
        helmet_color_pixels = cv2.countNonZero(all_helmet_colors)
        color_ratio = helmet_color_pixels / float(total_pixels)
        skin_pixels = cv2.countNonZero(skin_mask)
        hair_pixels = cv2.countNonZero(dark_hair_mask)
        skin_hair_ratio = (skin_pixels + hair_pixels) / float(total_pixels)

        # Bônus de superfície lisa só quando há presença clara de cor de capacete (evita parede cinza/calvície)
        smooth_bonus = 0.22 if (std_dev < 32.0 and color_ratio >= 0.18 and skin_hair_ratio <= 0.45) else 0.0

        if skin_hair_ratio > 0.60:
            score = max(0.0, color_ratio - 0.15)
        else:
            score = color_ratio + smooth_bonus

        score = float(np.clip(score, 0.0, 1.0))
        has_helmet = score >= self.helmet_threshold
        return has_helmet, score, (hx1, hy1, hx2, hy2)

    def _analyze_uniform_rf(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        kps: np.ndarray,
    ) -> Tuple[bool, float, Tuple[int, int, int, int]]:
        """Analisa o tronco buscando o Uniforme RF (ombro laranja + faixa refletiva cinza/prata)."""
        x1, y1, x2, y2 = box
        frame_h, frame_w = frame.shape[:2]
        height = y2 - y1
        box_w = x2 - x1

        sl = self._get_keypoint(kps, SHOULDER_L)
        sr = self._get_keypoint(kps, SHOULDER_R)
        hl = self._get_keypoint(kps, HIP_L)
        hr = self._get_keypoint(kps, HIP_R)

        # Limite superior: linha dos ombros (alcança a pala laranja superior mesmo de lado/costas)
        if sl and sr:
            top = max(y1 + int(height * 0.09), min(sl[1], sr[1]) - int(height * 0.10))
            mid_x = (sl[0] + sr[0]) // 2
            half_w = max(abs(sr[0] - sl[0]) // 2 + int(height * 0.09), int(box_w * 0.38), int(height * 0.12))
            left = mid_x - half_w
            right = mid_x + half_w
        elif sl or sr:
            pt = sl or sr
            top = max(y1 + int(height * 0.09), pt[1] - int(height * 0.10))
            half_w = max(int(height * 0.18), int(box_w * 0.38))
            left = pt[0] - half_w
            right = pt[0] + half_w
        else:
            top = y1 + int(height * 0.15)
            left = x1 + 4
            right = x2 - 4

        # Limite inferior: quadris ou proporção de tronco
        if hl and hr:
            bottom = max(hl[1], hr[1])
        elif hl or hr:
            bottom = (hl or hr)[1]
        else:
            bottom = top + int(height * 0.45)

        tx1 = max(0, left)
        ty1 = max(0, top)
        tx2 = min(frame_w - 1, right)
        ty2 = min(frame_h - 1, bottom)

        if tx2 <= tx1 or ty2 <= ty1:
            return False, 0.0, (x1, y1 + 30, x2, y2 - 30)

        torso_roi = frame[ty1:ty2, tx1:tx2]
        if torso_roi.size == 0:
            return False, 0.0, (tx1, ty1, tx2, ty2)

        torso_h, torso_w = torso_roi.shape[:2]
        hsv = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2HSV)

        # 1. Parte superior do peito/ombros (42% superiores do tronco onde fica a pala laranja fluorescente)
        upper_hsv = hsv[: max(1, int(torso_h * 0.42)), :]
        upper_pixels = upper_hsv.shape[0] * upper_hsv.shape[1]

        # Laranja fluorescente de seguranca (S>=135 exclui pele, camisas vinho/marrons e vestidos bege/amarelos)
        lower_orange1 = np.array([2, 135, 95], dtype=np.uint8)
        upper_orange1 = np.array([22, 255, 255], dtype=np.uint8)
        orange_mask = cv2.inRange(upper_hsv, lower_orange1, upper_orange1)

        # Fechamento morfológico para conectar o bloco laranja
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        orange_closed = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_ratio = cv2.countNonZero(orange_closed) / float(max(1, upper_pixels))

        # 2. Faixa refletiva horizontal cinza/prata de alta visibilidade
        lower_reflec = np.array([0, 0, 130], dtype=np.uint8)
        upper_reflec = np.array([180, 55, 255], dtype=np.uint8)
        reflec_mask = cv2.inRange(hsv, lower_reflec, upper_reflec)
        reflec_ratio = cv2.countNonZero(reflec_mask) / float(max(1, torso_h * torso_w))

        # 3. Tecido Cáqui / Bege / Cinza-cáqui RF Classe 2
        lower_khaki = np.array([10, 12, 55], dtype=np.uint8)
        upper_khaki = np.array([38, 110, 220], dtype=np.uint8)
        khaki_mask = cv2.inRange(hsv, lower_khaki, upper_khaki)
        khaki_ratio = cv2.countNonZero(khaki_mask) / float(max(1, torso_h * torso_w))

        # O Uniforme RF é confirmado pela camisa laranja total OU pela pala laranja + corpo cáqui/refletivo
        is_uniform = False
        score = 0.0

        if orange_ratio >= self.uniform_rf_threshold:
            score = round(min(1.0, orange_ratio * 2.5 + (0.2 if reflec_ratio > 0.025 else 0.0)), 3)
            is_uniform = True
        elif orange_ratio >= 0.035 and (reflec_ratio >= 0.025 or khaki_ratio >= 0.10):
            score = round(min(1.0, 0.65 + orange_ratio + reflec_ratio), 3)
            is_uniform = True

        return is_uniform, score, (tx1, ty1, tx2, ty2)

    def _analyze_gloves(
        self,
        frame: np.ndarray,
        wrist_pt: Tuple[int, int],
        elbow_pt: Optional[Tuple[int, int]],
        shoulder_w: int,
    ) -> Tuple[Optional[bool], float, Tuple[int, int, int, int]]:
        """Analisa a região das mãos para distinguir luvas (vaqueta/alta vis.) vs pele descoberta.

        Retorna ``None`` quando a mão está oclusa (sobre manga laranja, calça escura ou sombra),
        evitando tanto falsos positivos de ``gloves`` quanto falsos alertas de ``no_gloves``.
        """
        frame_h, frame_w = frame.shape[:2]
        wx, wy = wrist_pt

        if elbow_pt:
            ex, ey = elbow_pt
            dx = wx - ex
            dy = wy - ey
            length = max(1.0, np.hypot(dx, dy))
            hx = int(wx + (dx / length) * 14)
            hy = int(wy + (dy / length) * 14)
        else:
            hx, hy = wx, wy

        hand_radius = max(11, int(shoulder_w * 0.14)) if shoulder_w > 0 else 16
        hx1 = max(0, hx - hand_radius)
        hy1 = max(0, hy - hand_radius)
        hx2 = min(frame_w - 1, hx + hand_radius)
        hy2 = min(frame_h - 1, hy + hand_radius)

        if hx2 <= hx1 or hy2 <= hy1:
            return None, 0.0, (wx - 10, wy - 10, wx + 10, wy + 10)

        hand_roi = frame[hy1:hy2, hx1:hx2]
        if hand_roi.size == 0:
            return None, 0.0, (hx1, hy1, hx2, hy2)

        total_pixels = float(max(1, hand_roi.shape[0] * hand_roi.shape[1]))
        hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)

        # 1. Filtro de oclusão por manga laranja RF ou calça escura / sombra profunda
        orange1 = cv2.inRange(hsv, np.array([0, 165, 70]), np.array([22, 255, 255]))
        orange2 = cv2.inRange(hsv, np.array([165, 165, 70]), np.array([180, 255, 255]))
        orange_sleeve_ratio = cv2.countNonZero(orange1 | orange2) / total_pixels
        dark_ratio = float(np.mean(hsv[:, :, 2] <= 42))
        if orange_sleeve_ratio >= 0.35 or dark_ratio >= 0.65:
            return None, 0.0, (hx1, hy1, hx2, hy2)

        # 2. Luva de vaqueta (couro claro/branco/cinza: baixa saturação e brilho médio-alto)
        vaqueta_mask = (hsv[:, :, 1] <= 58) & (hsv[:, :, 2] >= 100)
        vaqueta_ratio = float(np.mean(vaqueta_mask))

        # 3. Luva colorida / isolante / alta visibilidade (amarela, verde, azul)
        hivis_mask = cv2.inRange(hsv, np.array([24, 75, 50]), np.array([135, 255, 255]))
        hivis_ratio = cv2.countNonZero(hivis_mask) / total_pixels
        glove_ratio = vaqueta_ratio + hivis_ratio

        # 4. Pele humana descoberta (saturação >= 58 exclui couro claro de vaqueta)
        skin_mask = cv2.inRange(hsv, np.array([0, 58, 45]), np.array([22, 195, 235]))
        skin_ratio = cv2.countNonZero(skin_mask) / total_pixels

        if ((vaqueta_ratio >= 0.34) or (hivis_ratio >= 0.25)) and glove_ratio > skin_ratio * 0.90:
            return True, round(min(0.95, 0.55 + glove_ratio), 3), (hx1, hy1, hx2, hy2)
        if skin_ratio >= max(self.gloves_skin_threshold, 0.40) and glove_ratio < 0.15:
            return False, round(min(0.95, 0.45 + skin_ratio), 3), (hx1, hy1, hx2, hy2)

        return None, 0.0, (hx1, hy1, hx2, hy2)

    def detect(self, frame: np.ndarray, skip_frames: int = 1) -> List[Detection]:
        """Executa a inferência de Pose e avaliação anatômica de EPIs."""
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]

        try:
            # Resolução dinâmica: se for close-up/cabine, roda com 768px para maior precisão
            imgsz_target = self.imgsz

            if self.track:
                results = self.model.track(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    imgsz=imgsz_target,
                    device=self.device,
                    persist=True,
                    verbose=False,
                )
            else:
                results = self.model(
                    source=frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    imgsz=imgsz_target,
                    device=self.device,
                    verbose=False,
                )

            if not results or len(results) == 0:
                return []

            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = (
                result.boxes.id.int().cpu().numpy()
                if result.boxes.id is not None
                else [None] * len(boxes)
            )
            keypoints_data = (
                result.keypoints.data.cpu().numpy()
                if result.keypoints is not None
                else None
            )

            detections: List[Detection] = []

            # Suprime duplicatas aninhadas (ex.: caixa de tronco sobreposta a pessoa inteira)
            raw_areas = [max(1, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
            suppressed: set[int] = set()
            for i in range(len(boxes)):
                if i in suppressed:
                    continue
                ax1, ay1, ax2, ay2 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if j in suppressed:
                        continue
                    bx1, by1, bx2, by2 = boxes[j]
                    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        inter = (ix2 - ix1) * (iy2 - iy1)
                        if inter / min(raw_areas[i], raw_areas[j]) >= 0.58:
                            suppressed.add(j if confs[i] >= confs[j] else i)

            for i, box in enumerate(boxes):
                if i in suppressed:
                    continue
                x1, y1, x2, y2 = map(int, box)
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)

                # Descarte de falsos positivos: proporção achatada ou artefato de lança no topo da tela
                if (box_w / float(box_h)) > 2.6:
                    continue

                conf = round(float(confs[i]), 3)
                kps = keypoints_data[i] if keypoints_data is not None and i < len(keypoints_data) else None

                # Mantém a pessoa se houver keypoints suficientes ou confiança sólida (evita queda do track em oclusão breve)
                if kps is not None:
                    valid_kps = sum(1 for kp in kps if kp[2] >= 0.30 and (kp[0] > 0 or kp[1] > 0))
                    if valid_kps < 2 and conf < 0.45:
                        continue
                    if y1 <= 10 and valid_kps < 4 and box_w > box_h * 0.85:
                        continue

                track_id = int(track_ids[i]) if track_ids[i] is not None else -1

                if kps is None:
                    if self.only_company_workers:
                        detections.append(Detection(
                            class_name="civil",
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            track_id=track_id,
                        ))
                    else:
                        detections.append(Detection(
                            class_name="Person",
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            track_id=track_id,
                        ))
                    continue

                sl = self._get_keypoint(kps, SHOULDER_L)
                sr = self._get_keypoint(kps, SHOULDER_R)
                shoulder_w = abs(sr[0] - sl[0]) if (sl and sr) else int(box_w * 0.5)

                # Análise das características de EPI
                has_uniform, uniform_score, torso_box = self._analyze_uniform_rf(frame, (x1, y1, x2, y2), kps)
                has_helmet, helmet_score, head_box = self._analyze_helmet(frame, (x1, y1, x2, y2), kps)

                # Avaliação de contexto de cabine (somente close-up real de motorista)
                is_cabin_person = self.is_cabin or (
                    (box_h / float(frame_h)) > 0.62
                    and (box_w / float(frame_w)) > 0.25
                    and (y2 >= frame_h * 0.85)
                )

                # Critério de Funcionário da Empresa vs Pedestre/Civil na rua
                is_company_worker = has_uniform or (has_helmet and helmet_score >= 0.48) or is_cabin_person

                # Se for civil na rua (sem uniforme RF e sem capacete): NÃO audita nem gera infração!
                if self.only_company_workers and not is_company_worker:
                    detections.append(Detection(
                        class_name="civil",
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        track_id=track_id,
                    ))
                    continue

                # 1. Registro do Trabalhador da Empresa
                detections.append(Detection(
                    class_name="Person",
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                ))

                # 2. Avaliação de Capacete
                if has_helmet:
                    detections.append(Detection(
                        class_name="helmet",
                        confidence=helmet_score,
                        bbox=head_box,
                        track_id=track_id,
                    ))
                else:
                    detections.append(Detection(
                        class_name="no_helmet",
                        confidence=round(1.0 - helmet_score, 3),
                        bbox=head_box,
                        track_id=track_id,
                    ))

                # 3. Avaliação de Uniforme RF (e compatibilidade Colete)
                if has_uniform:
                    detections.append(Detection(
                        class_name="uniform_rf",
                        confidence=uniform_score,
                        bbox=torso_box,
                        track_id=track_id,
                    ))
                    detections.append(Detection(
                        class_name="vest",
                        confidence=uniform_score,
                        bbox=torso_box,
                        track_id=track_id,
                    ))
                else:
                    # Trabalhador descaracterizado (em cabine ou no canteiro)
                    detections.append(Detection(
                        class_name="no_uniform_rf",
                        confidence=0.85,
                        bbox=torso_box,
                        track_id=track_id,
                    ))
                    detections.append(Detection(
                        class_name="no_vest",
                        confidence=0.85,
                        bbox=torso_box,
                        track_id=track_id,
                    ))

                # 4. Avaliação de Luvas (Pulsos confiáveis e não oclusos atrás do tronco)
                wl = self._get_keypoint(kps, WRIST_L, min_conf=0.50)
                wr = self._get_keypoint(kps, WRIST_R, min_conf=0.50)
                el = self._get_keypoint(kps, ELBOW_L)
                er = self._get_keypoint(kps, ELBOW_R)
                has_face = any(self._get_keypoint(kps, idx, min_conf=0.30) is not None for idx in (NOSE, EYE_L, EYE_R))
                body_cx = (x1 + x2) / 2.0

                for wrist, elbow in [(wl, el), (wr, er)]:
                    if not wrist:
                        continue
                    # Trabalhador de costas com pulso no meio do tronco -> mão oclusa na frente do corpo/painel
                    if (
                        not has_face
                        and abs(wrist[0] - body_cx) < box_w * 0.22
                        and (y1 + box_h * 0.18) <= wrist[1] <= (y1 + box_h * 0.56)
                    ):
                        continue
                    has_gloves, glove_conf, hand_box = self._analyze_gloves(
                        frame, wrist, elbow, shoulder_w
                    )
                    if has_gloves is True:
                        detections.append(Detection(
                            class_name="gloves",
                            confidence=glove_conf,
                            bbox=hand_box,
                            track_id=track_id,
                        ))
                    elif has_gloves is False:
                        detections.append(Detection(
                            class_name="no_gloves",
                            confidence=glove_conf,
                            bbox=hand_box,
                            track_id=track_id,
                        ))

            return detections

        except Exception as e:
            logger.error(f"Erro na inferência do PoseEPIDetector: {e}", exc_info=True)
            return []

    def reset_tracking(self) -> None:
        """Reseta o rastreamento interno do ByteTrack para um novo vídeo/stream."""
        try:
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    delattr(self.model.predictor, "trackers")
            logger.debug("🔄 Tracking do modelo Pose resetado.")
        except Exception:
            pass

    @property
    def confidence(self) -> float:
        return self._confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self._confidence = float(value)

    @property
    def class_names(self) -> Dict[int, str]:
        return {
            0: "Person",
            1: "helmet",
            2: "no_helmet",
            3: "uniform_rf",
            4: "no_uniform_rf",
            5: "gloves",
            6: "no_gloves",
            7: "vest",
            8: "no_vest",
        }

