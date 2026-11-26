"""Armazenamento local de evidências visuais de infrações em disco."""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.core.domain.violation import Violation

logger = logging.getLogger(__name__)

_UNSAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_segment(value: str | None, fallback: str = "desconhecido") -> str:
    """Sanitiza um segmento de caminho/arquivo para impedir path traversal."""
    raw = str(value or "").strip()
    cleaned = _UNSAFE_SEGMENT_RE.sub("_", raw)
    while ".." in cleaned:
        cleaned = cleaned.replace("..", "_")
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


class FileAlertStorage:
    """Implementa IAlertStorage organizando arquivos por Placa e Data."""

    def __init__(self, base_dir: str | Path = "alertas", image_quality: int = 85):
        self.base_dir = Path(base_dir)
        self.image_quality = max(1, min(100, image_quality))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._images_saved_count = 0
        self._lock = threading.Lock()

    def save_image(self, frame: np.ndarray, violation: Violation) -> str | None:
        """Salva a imagem no disco em subpastas ou na raiz de alertas."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            violation_slug = _sanitize_segment(violation.violation_type, "violacao")
            track_str = f"_id{int(violation.track_id)}" if violation.track_id is not None else ""

            meta = violation.metadata
            if meta and (meta.plate or meta.record_date):
                plate_dir = _sanitize_segment(meta.plate, "DESCONHECIDO")
                date_dir = _sanitize_segment(meta.record_date, datetime.now().strftime("%Y-%m-%d"))
                target_dir = self.base_dir / plate_dir / date_dir
                v_time = _sanitize_segment(meta.video_time_str.replace(":", "-"), "00-00-00")
                v_name = _sanitize_segment(Path(meta.video_name or "frame").stem, "frame")
                filename = f"alerta_{v_name}_{v_time}_{violation_slug}{track_str}.jpg"
            else:
                target_dir = self.base_dir
                filename = f"alerta_{timestamp}_{violation_slug}{track_str}.jpg"

            base_resolved = self.base_dir.resolve()
            filepath = target_dir / filename
            if not filepath.resolve().is_relative_to(base_resolved):
                logger.error("❌ Caminho de evidência rejeitado por segurança: %s", filepath)
                return None

            target_dir.mkdir(parents=True, exist_ok=True)

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success = cv2.imwrite(str(filepath), frame, encode_params)

            if success:
                with self._lock:
                    self._images_saved_count += 1
                logger.info(f"💾 Evidência salva: {filepath}")
                return str(filepath.resolve())
            else:
                logger.error(f"❌ cv2.imwrite falhou ao salvar: {filepath}")
                return None

        except Exception as e:
            logger.error(f"❌ Erro ao salvar imagem de alerta: {e}")
            return None

    @property
    def total_saved(self) -> int:
        return self._images_saved_count
