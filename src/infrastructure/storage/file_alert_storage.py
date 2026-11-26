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

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str | None, fallback: str) -> str:
    """Sanitiza um nome usado como pasta/arquivo.

    A placa/data vêm de nomes de pasta e do ``camera_id`` informado na API/câmeras; sem
    sanitização um valor como ``../../Windows`` gravaria fora da pasta de alertas, e
    caracteres como ``:`` quebram caminhos no Windows.
    """
    cleaned = _UNSAFE_CHARS.sub("_", str(value or "")).strip("._")
    return cleaned[:80] or fallback


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
            violation_slug = _safe_component(violation.violation_type, "infracao")
            track_str = f"_id{violation.track_id}" if violation.track_id is not None else ""

            meta = violation.metadata
            if meta and (meta.plate or meta.record_date):
                target_dir = (
                    self.base_dir
                    / _safe_component(meta.plate, "SEM_PLACA")
                    / _safe_component(meta.record_date, datetime.now().strftime("%Y-%m-%d"))
                )
                v_time = _safe_component(meta.video_time_str.replace(":", "-"), "00-00-00")
                v_name = _safe_component(Path(meta.video_name).stem, "video")
                filename = f"alerta_{v_name}_{v_time}_{violation_slug}{track_str}.jpg"
            else:
                target_dir = self.base_dir
                filename = f"alerta_{timestamp}_{violation_slug}{track_str}.jpg"

            target_dir.mkdir(parents=True, exist_ok=True)
            filepath = target_dir / filename
            # Não sobrescreve evidências anteriores com o mesmo nome (ex.: reprocessamento)
            counter = 1
            while filepath.exists():
                filepath = target_dir / f"{Path(filename).stem}_{counter}.jpg"
                counter += 1

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
