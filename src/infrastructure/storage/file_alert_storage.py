"""Armazenamento local de evidências visuais de infrações em disco."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from src.core.domain.violation import Violation

logger = logging.getLogger(__name__)


class FileAlertStorage:
    """Implementa IAlertStorage organizando arquivos por Placa e Data."""

    def __init__(self, base_dir: str | Path = "alertas", image_quality: int = 85):
        self.base_dir = Path(base_dir)
        self.image_quality = max(1, min(100, image_quality))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._images_saved_count = 0

    def save_image(self, frame: np.ndarray, violation: Violation) -> Optional[str]:
        """Salva a imagem no disco em subpastas ou na raiz de alertas."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            violation_slug = violation.violation_type.replace(" ", "_")
            track_str = f"_id{violation.track_id}" if violation.track_id is not None else ""

            meta = violation.metadata
            if meta and (meta.plate or meta.record_date):
                target_dir = self.base_dir / meta.plate / meta.record_date
                v_time = meta.video_time_str.replace(":", "-")
                v_name = Path(meta.video_name).stem
                filename = f"alerta_{v_name}_{v_time}_{violation_slug}{track_str}.jpg"
            else:
                target_dir = self.base_dir
                filename = f"alerta_{timestamp}_{violation_slug}{track_str}.jpg"

            target_dir.mkdir(parents=True, exist_ok=True)
            filepath = target_dir / filename

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            success = cv2.imwrite(str(filepath), frame, encode_params)

            if success:
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
