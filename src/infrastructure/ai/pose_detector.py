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
        hy1 = top
        hy2 = min(frame_h - 1, top + max(span, 15))

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
        lower_skin = np.array([0, 35, 60], dtype=np.uint8)
        upper_skin = np.array([25, 175, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin) & mask

        # 2. Filtro de cabelo escuro / texturizado
        dark_hair_mask = (hsv[:, :, 2] < 45) & mask

        # 3. Superfície lisa de plástico (desvio padrão de brilho baixo na calota)
        masked_gray = gray[mask > 0]
        std_dev = np.std(masked_gray) if len(masked_gray) > 0 else 999.0
        smooth_bonus = 0.25 if std_dev < 32.0 else 0.0

        # 4. Cores industriais de capacete (Branco, Amarelo, Laranja, Azul, Vermelho, Verde)
        # Branco industrial (alto V, baixo S)
        white_mask = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 165) & mask

        # Amarelo / Laranja industrial
        yellow_orange_mask = (
            cv2.inRange(hsv, np.array([10, 60, 70]), np.array([38, 255, 255])) & mask
        )

        # Azul industrial
        blue_mask = (
            cv2.inRange(hsv, np.array([90, 60, 60]), np.array([130, 255, 255])) & mask
        )

        # Vermelho industrial
        red1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([8, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 70, 70]), np.array([180, 255, 255]))
        red_mask = (red1 | red2) & mask

        helmet_color_pixels = cv2.countNonZero(white_mask | yellow_orange_mask | blue_mask | red_mask)
        skin_pixels = cv2.countNonZero(skin_mask)
        hair_pixels = cv2.countNonZero(dark_hair_mask)

        # Se a cabeça for dominada por pele ou cabelo escuro sem brilho liso
        if (skin_pixels + hair_pixels) / float(total_pixels) > 0.60:
            score = max(0.0, (helmet_color_pixels / float(total_pixels)) - 0.15)
        else:
            score = (helmet_color_pixels / float(total_pixels)) + smooth_bonus

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

        sl = self._get_keypoint(kps, SHOULDER_L)
        sr = self._get_keypoint(kps, SHOULDER_R)
        hl = self._get_keypoint(kps, HIP_L)
        hr = self._get_keypoint(kps, HIP_R)

        # Limite superior: linha dos ombros
        if sl and sr:
            top = min(sl[1], sr[1]) - int(height * 0.03)
            left = min(sl[0], sr[0]) - int(height * 0.08)
            right = max(sl[0], sr[0]) + int(height * 0.08)
        elif sl or sr:
            pt = sl or sr
            top = pt[1] - int(height * 0.03)
            left = pt[0] - int(height * 0.20)
            right = pt[0] + int(height * 0.20)
        else:
            top = y1 + int(height * 0.18)
            left = x1 + 5
            right = x2 - 5

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

        # 1. Parte superior do peito/ombros (primeiros 50% do tronco) onde fica o bloco laranja
        upper_hsv = hsv[: max(1, int(torso_h * 0.55)), :]
        upper_pixels = upper_hsv.shape[0] * upper_hsv.shape[1]

        # Faixa de Laranja Fluorescente da farda RF
        lower_orange1 = np.array([4, 70, 70], dtype=np.uint8)
        upper_orange1 = np.array([24, 255, 255], dtype=np.uint8)
        orange_mask = cv2.inRange(upper_hsv, lower_orange1, upper_orange1)

        # Fechamento morfológico para conectar o bloco laranja
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        orange_closed = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_ratio = cv2.countNonZero(orange_closed) / float(max(1, upper_pixels))

        # 2. Faixa refletiva horizontal cinza/prata de alta visibilidade
        # Prata refletivo: saturação baixa e brilho médio/alto
        lower_reflec = np.array([0, 0, 130], dtype=np.uint8)
        upper_reflec = np.array([180, 55, 255], dtype=np.uint8)
        reflec_mask = cv2.inRange(hsv, lower_reflec, upper_reflec)
        reflec_ratio = cv2.countNonZero(reflec_mask) / float(max(1, torso_h * torso_w))

        # 3. Tecido Cáqui / Bege RF Classe 2
        lower_khaki = np.array([14, 20, 60], dtype=np.uint8)
        upper_khaki = np.array([35, 130, 220], dtype=np.uint8)
        khaki_mask = cv2.inRange(hsv, lower_khaki, upper_khaki)
        khaki_ratio = cv2.countNonZero(khaki_mask) / float(max(1, torso_h * torso_w))

        # O Uniforme RF é confirmado pela presença do bloco superior laranja
        # e pela faixa refletiva ou pelo contraste com o corpo cáqui
        is_uniform = False
        score = 0.0

        if orange_ratio >= self.uniform_rf_threshold:
            # Laranja presente no ombro/peito
            score = round(min(1.0, orange_ratio * 2.5 + (0.2 if reflec_ratio > 0.03 else 0.0)), 3)
            is_uniform = True
        elif orange_ratio >= 0.07 and reflec_ratio >= 0.04:
            # Laranja parcial + faixa refletiva nítida
            score = round(min(1.0, 0.65 + reflec_ratio), 3)
            is_uniform = True
        elif reflec_ratio >= 0.08 and khaki_ratio >= 0.20:
            # Faixa refletiva clara com tecido cáqui operacional
            score = round(min(1.0, 0.60 + khaki_ratio * 0.5), 3)
            is_uniform = True

        return is_uniform, score, (tx1, ty1, tx2, ty2)

    def _analyze_gloves(
        self,
        frame: np.ndarray,
        wrist_pt: Tuple[int, int],
        elbow_pt: Optional[Tuple[int, int]],
        shoulder_w: int,
    ) -> Tuple[bool, float, Tuple[int, int, int, int]]:
        """Analisa a região das mãos/pulsos para verificar luvas vs pele descoberta."""
        frame_h, frame_w = frame.shape[:2]
        wx, wy = wrist_pt

        # Vetor de projeção do antebraço até o centro da mão
        if elbow_pt:
            ex, ey = elbow_pt
            dx = wx - ex
            dy = wy - ey
            length = max(1.0, np.hypot(dx, dy))
            # Mão projetada além do pulso
            hx = int(wx + (dx / length) * 16)
            hy = int(wy + (dy / length) * 16)
        else:
            hx, hy = wx, wy

        hand_radius = max(12, int(shoulder_w * 0.14)) if shoulder_w > 0 else 18
        hx1 = max(0, hx - hand_radius)
        hy1 = max(0, hy - hand_radius)
        hx2 = min(frame_w - 1, hx + hand_radius)
        hy2 = min(frame_h - 1, hy + hand_radius)

        if hx2 <= hx1 or hy2 <= hy1:
            return False, 0.0, (wx - 10, wy - 10, wx + 10, wy + 10)

        hand_roi = frame[hy1:hy2, hx1:hx2]
        if hand_roi.size == 0:
            return False, 0.0, (hx1, hy1, hx2, hy2)

        total_pixels = hand_roi.shape[0] * hand_roi.shape[1]
        hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)

        # Máscara de pele humana (Bare skin)
        lower_skin = np.array([0, 35, 60], dtype=np.uint8)
        upper_skin = np.array([25, 175, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = cv2.countNonZero(skin_mask) / float(max(1, total_pixels))

        # Se a mão tiver alta proporção de tom de pele, está sem luva
        if skin_ratio >= self.gloves_skin_threshold:
            has_gloves = False
            conf = round(skin_ratio, 3)
        else:
            has_gloves = True
            conf = round(1.0 - skin_ratio, 3)

        return has_gloves, conf, (hx1, hy1, hx2, hy2)

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

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)

                # Descarte de falsos positivos: proporção achatada (reflexos no chão)
                if (box_w / float(box_h)) > 2.6:
                    continue

                kps = keypoints_data[i] if keypoints_data is not None and i < len(keypoints_data) else None

                # Verifica se há keypoints válidos suficientes para uma pessoa real
                if kps is not None:
                    valid_kps = sum(1 for kp in kps if kp[2] >= 0.30 and (kp[0] > 0 or kp[1] > 0))
                    if valid_kps < 3:
                        continue

                track_id = int(track_ids[i]) if track_ids[i] is not None else -1
                conf = round(float(confs[i]), 3)

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

                # Avaliação de contexto de cabine (motorista em close-up)
                is_cabin_person = self.is_cabin or ((box_h / float(frame_h)) > 0.40 and (y2 >= frame_h * 0.85))

                # Critério de Funcionário da Empresa vs Pedestre/Civil na rua
                is_company_worker = has_uniform or has_helmet or is_cabin_person

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

                # 4. Avaliação de Luvas (Pulsos)
                wl = self._get_keypoint(kps, WRIST_L)
                wr = self._get_keypoint(kps, WRIST_R)
                el = self._get_keypoint(kps, ELBOW_L)
                er = self._get_keypoint(kps, ELBOW_R)

                for wrist, elbow in [(wl, el), (wr, er)]:
                    if wrist:
                        has_gloves, glove_conf, hand_box = self._analyze_gloves(
                            frame, wrist, elbow, shoulder_w
                        )
                        if has_gloves:
                            detections.append(Detection(
                                class_name="gloves",
                                confidence=glove_conf,
                                bbox=hand_box,
                                track_id=track_id,
                            ))
                        else:
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

