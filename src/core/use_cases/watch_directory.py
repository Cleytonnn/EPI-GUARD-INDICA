"""Caso de uso: Monitoramento contínuo de diretório por novas gravações (Watcher)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional
from src.core.use_cases.process_batch import ProcessBatchUseCase

logger = logging.getLogger(__name__)


class WatchDirectoryUseCase:
    """Monitora continuamente o diretório por novas gravações finalizadas."""

    def __init__(self, batch_uc: ProcessBatchUseCase):
        self.batch_uc = batch_uc
        self._running = False

    def execute(
        self,
        input_dir: Path | str,
        interval_seconds: int = 30,
        skip_frames: int = 3,
    ) -> None:
        """Inicia o loop contínuo de monitoramento."""
        dir_obj = Path(input_dir).resolve()
        self._running = True

        logger.info("=" * 60)
        logger.info("👀 MODO WATCHER ATIVADO (MONITORAMENTO CONTÍNUO)")
        logger.info(f"   Diretório monitorado: {dir_obj}")
        logger.info(f"   Intervalo de checagem: a cada {interval_seconds}s")
        logger.info("   Pressione Ctrl+C para encerrar.")
        logger.info("=" * 60)

        try:
            while self._running:
                videos = self.batch_uc.find_videos(dir_obj)
                pending = [
                    v for v in videos
                    if not self.batch_uc.processed_repo.is_processed(v) and self._is_file_ready(v)
                ]

                if pending:
                    logger.info(f"🔔 {len(pending)} novo(s) vídeo(s) detectado(s) para análise!")
                    for v in pending:
                        self.batch_uc.process_video_uc.execute(
                            video_path=v,
                            root_dir=dir_obj,
                            skip_frames=skip_frames,
                        )
                else:
                    logger.debug(f"💤 Nenhum vídeo novo. Aguardando {interval_seconds}s...")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("🛑 Monitoramento contínuo encerrado pelo usuário.")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def _is_file_ready(self, file_path: Path, wait_check: float = 1.0) -> bool:
        """Garante que a gravação já foi concluída e o arquivo está estável."""
        try:
            size1 = file_path.stat().st_size
            time.sleep(wait_check)
            size2 = file_path.stat().st_size
            return size1 == size2 and size1 > 0
        except Exception:
            return False
