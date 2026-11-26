"""Caso de uso: Processamento em lote de gravações organizadas por pastas."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.interfaces.processed_repo import IProcessedVideosRepository
from src.core.use_cases.process_video_file import ProcessVideoFileUseCase, VideoProcessingResult

logger = logging.getLogger(__name__)


class ProcessBatchUseCase:
    """Orquestra a varredura e processamento em lote de múltiplos arquivos de vídeo."""

    def __init__(
        self,
        process_video_uc: ProcessVideoFileUseCase,
        processed_repo: IProcessedVideosRepository,
        supported_extensions: list[str] | None = None,
    ):
        self.process_video_uc = process_video_uc
        self.processed_repo = processed_repo
        self.supported_extensions = set(
            ext.lower() for ext in (supported_extensions or [".265", ".h265", ".hevc", ".mp4", ".avi", ".mkv"])
        )

    def find_videos(self, input_dir: Path | str) -> list[Path]:
        """Varre recursivamente o diretório por arquivos suportados."""
        dir_obj = Path(input_dir).resolve()
        if not dir_obj.exists():
            logger.warning(f"⚠️ Diretório de entrada não existe: {dir_obj}")
            return []

        videos = [p for p in dir_obj.rglob("*") if p.is_file() and p.suffix.lower() in self.supported_extensions]
        videos.sort()
        return videos

    def execute(
        self,
        input_dir: Path | str,
        skip_frames: int = 3,
        force_reprocess: bool = False,
    ) -> list[VideoProcessingResult]:
        """Executa a análise em todos os vídeos pendentes do diretório."""
        all_videos = self.find_videos(input_dir)
        logger.info(f"🔍 {len(all_videos)} arquivo(s) de vídeo encontrado(s) na pasta.")

        pending_videos = []
        for v in all_videos:
            if force_reprocess or not self.processed_repo.is_processed(v):
                pending_videos.append(v)
            else:
                logger.debug(f"⏭️ Vídeo já processado anteriormente: {v.name}")

        logger.info(f"📋 {len(pending_videos)} vídeo(s) pendente(s) para processamento.")

        results: list[VideoProcessingResult] = []
        for idx, video_path in enumerate(pending_videos, 1):
            logger.info(f"\n[{idx}/{len(pending_videos)}] Iniciando análise...")
            try:
                res = self.process_video_uc.execute(
                    video_path=video_path,
                    root_dir=input_dir,
                    skip_frames=skip_frames,
                )
            except Exception as exc:  # um vídeo com defeito não pode interromper o lote inteiro
                logger.error(f"❌ Falha ao processar {video_path.name}: {exc}", exc_info=True)
                res = VideoProcessingResult(
                    success=False,
                    video_path=str(video_path),
                    plate="",
                    record_date="",
                    duration_seconds=0.0,
                    elapsed_seconds=0.0,
                    violations_count=0,
                    violations=[],
                    error=str(exc),
                )
            results.append(res)

        self._log_summary(results)
        return results

    def _log_summary(self, results: list[VideoProcessingResult]) -> None:
        total = len(results)
        failed = sum(1 for r in results if not r.success)
        violations = sum(r.violations_count for r in results)
        total_time = sum(r.elapsed_seconds for r in results)

        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMO GERAL DO PROCESSAMENTO DE VÍDEOS")
        logger.info("=" * 60)
        logger.info(f"   Total de vídeos analisados: {total} (com falha: {failed})")
        logger.info(f"   Total de infrações de EPI encontradas: {violations}")
        logger.info(f"   Tempo total de processamento: {total_time:.1f}s")
        logger.info("=" * 60)
