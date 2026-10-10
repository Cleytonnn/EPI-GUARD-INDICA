"""Adaptador de inferência baseado no Ultralytics YOLO."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from ultralytics import YOLO

from src.core.domain.detection import Detection
from src.core.interfaces.detector import IDetector

logger = logging.getLogger(__name__)


class YOLODetector(IDetector):
    """Implementa IDetector utilizando modelos Ultralytics YOLO."""

    def __init__(
        self,
        model_path: str | Path = "models/best.pt",
        confidence: float = 0.4,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        track: bool = True,
    ):
        self.model_path = str(model_path)
        self._confidence = float(confidence)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.track = bool(track)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self) -> None:
        logger.info(f"🤖 Carregando modelo YOLO em [{self.device.upper()}]: {self.model_path}")
        self.model = YOLO(self.model_path)
        self._class_names = self.model.names
        logger.info(f"🤖 Modelo carregado com sucesso. Classes ({len(self._class_names)}): {list(self._class_names.values())}")

    def detect(self, frame: np.ndarray, skip_frames: int = 1) -> List[Detection]:
        """Executa a inferência ou tracking no frame fornecido."""
        if frame is None or frame.size == 0:
            return []

        try:
            if self.track:
                results = self.model.track(
                    source=frame,
                    conf=self._confidence,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    persist=True,
                    verbose=False,
                )
            else:
                results = self.model.predict(
                    source=frame,
                    conf=self._confidence,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )

            if not results or len(results) == 0:
                return []

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return []

            detections: List[Detection] = []
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self._class_names.get(cls_id, f"class_{cls_id}")

                track_id: Optional[int] = None
                if box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())

                detections.append(Detection(
                    class_name=class_name,
                    confidence=round(conf, 3),
                    bbox=(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])),
                    track_id=track_id,
                ))

            return detections

        except Exception as e:
            logger.error(f"❌ Erro durante a inferência YOLO: {e}")
            return []

    def reset_tracking(self) -> None:
        """Reseta o rastreamento interno do ByteTrack para um novo vídeo/stream."""
        try:
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    self.model.predictor.trackers = None
            logger.debug("🔄 Tracking do modelo resetado.")
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
        return self._class_names
