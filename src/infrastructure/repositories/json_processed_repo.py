"""Repositório de histórico de vídeos processados baseado em JSON."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.interfaces.processed_repo import IProcessedVideosRepository

logger = logging.getLogger(__name__)


class JsonProcessedVideosRepository(IProcessedVideosRepository):
    """Implementa IProcessedVideosRepository persistindo em arquivo JSON local."""

    def __init__(self, db_path: str | Path = "videos_processados.json"):
        self.db_path = Path(db_path)
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar banco de vídeos processados: {e}")
                self.data = {}
        else:
            self.data = {}

    def _save(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar banco de vídeos processados: {e}")

    def is_processed(self, video_path: Path | str) -> bool:
        """Verifica se o arquivo já foi processado e seu mtime não foi modificado."""
        key = str(Path(video_path).resolve())
        if key not in self.data:
            return False

        try:
            current_mtime = Path(video_path).stat().st_mtime
            saved_mtime = self.data[key].get("mtime", 0.0)
            return current_mtime <= saved_mtime
        except Exception:
            return False

    def mark_processed(
        self,
        video_path: Path | str,
        metadata: VehicleMetadata,
        violations_count: int,
        duration_seconds: float,
    ) -> None:
        """Registra a conclusão da análise do arquivo de vídeo."""
        path_obj = Path(video_path).resolve()
        key = str(path_obj)
        try:
            mtime = path_obj.stat().st_mtime
            size = path_obj.stat().st_size
        except Exception:
            mtime = time.time()
            size = 0

        self.data[key] = {
            "placa": metadata.plate,
            "data": metadata.record_date,
            "video_name": metadata.video_name,
            "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mtime": mtime,
            "size_bytes": size,
            "duration_seconds": round(duration_seconds, 2),
            "violations_count": violations_count,
        }
        self._save()
