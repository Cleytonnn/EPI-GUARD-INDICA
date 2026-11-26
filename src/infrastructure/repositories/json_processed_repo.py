"""Repositório de histórico de vídeos processados baseado em JSON."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.interfaces.processed_repo import IProcessedVideosRepository

logger = logging.getLogger(__name__)


class JsonProcessedVideosRepository(IProcessedVideosRepository):
    """Implementa IProcessedVideosRepository persistindo em arquivo JSON local."""

    def __init__(self, db_path: str | Path = "videos_processados.json"):
        self.db_path = Path(db_path)
        self.data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.db_path.exists():
            try:
                with open(self.db_path, encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Erro ao carregar banco de vídeos processados: {e}")
                self.data = {}
        else:
            self.data = {}

    def _save(self) -> None:
        """Grava de forma atômica (arquivo temporário + replace).

        Antes o JSON era sobrescrito diretamente: se o programa fechasse no meio da gravação o
        arquivo ficava corrompido, era descartado no próximo carregamento e TODOS os vídeos
        voltavam a ser reprocessados.
        """
        tmp_name = None
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".videos_", suffix=".tmp", dir=self.db_path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_name, self.db_path)
            tmp_name = None
        except Exception as e:
            logger.error(f"❌ Erro ao salvar banco de vídeos processados: {e}")
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

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
        with self._lock:
            self._save()
