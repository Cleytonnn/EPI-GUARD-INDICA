"""Ponto de entrada para processamento em lote e monitor de gravações MP4/.265.

Mantém 100% de compatibilidade retroativa com a API anterior delegando
para os casos de uso da Clean Architecture.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.container import Container
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.presentation.cli.commands import (
    run_batch_command,
    run_single_file_command,
    run_watch_command,
)

logger = logging.getLogger(__name__)


def extract_metadata_from_path(video_path: Path | str, root_dir: Path | str | None = None) -> dict[str, str]:
    """Função utilitária retrocompatível para extração de metadados."""
    meta = VehicleMetadata.from_path(video_path, root_dir)
    return {
        "placa": meta.plate,
        "data": meta.record_date,
        "video_name": meta.video_name,
    }


class VideoBatchProcessor:
    """Fachada retrocompatível para o caso de uso de processamento em lote."""

    def __init__(
        self,
        input_dir: str | None = None,
        config_path: str | None = None,
        skip_frames: int | None = None,
        show_window: bool = False,
        report_csv: str | None = None,
        processed_db: str | None = None,
        force_reprocess: bool = False,
    ):
        from src.config.settings import AppSettings

        settings = AppSettings.load(config_path)

        if input_dir:
            settings.video_batch.input_dir = input_dir
        if skip_frames is not None:
            settings.video_batch.skip_frames = skip_frames
        if report_csv:
            settings.video_batch.report_csv = report_csv
        if processed_db:
            settings.video_batch.processed_db = processed_db

        self.container = Container(settings)
        self.input_dir = Path(settings.video_batch.input_dir)
        self.skip_frames = settings.video_batch.skip_frames
        self.force_reprocess = force_reprocess
        self.show_window = show_window

        # Aliases para compatibilidade
        self.detector = self.container.detector
        self.alerter = self.container.alert_storage
        self.db = self.container.processed_repo
        self.config = {
            "detection": {
                "model_path": settings.detection.model_path,
                "confidence": settings.detection.confidence,
            },
            "video_batch": {
                "input_dir": settings.video_batch.input_dir,
                "skip_frames": settings.video_batch.skip_frames,
                "watch_interval": settings.video_batch.watch_interval,
            },
        }

    def process_video(self, video_path: Path | str) -> dict[str, Any]:
        result = self.container.process_video_file_uc.execute(
            video_path=video_path,
            root_dir=self.input_dir,
            skip_frames=self.skip_frames,
        )
        return {
            "success": result.success,
            "video": result.video_path,
            "placa": result.plate,
            "data": result.record_date,
            "duration": result.duration_seconds,
            "elapsed_seconds": result.elapsed_seconds,
            "violations_count": result.violations_count,
            "violations": result.violations,
        }

    def process_all_pending(self) -> list[dict[str, Any]]:
        results = self.container.process_batch_uc.execute(
            input_dir=self.input_dir,
            skip_frames=self.skip_frames,
            force_reprocess=self.force_reprocess,
        )
        return [
            {
                "success": r.success,
                "video": r.video_path,
                "placa": r.plate,
                "data": r.record_date,
                "duration": r.duration_seconds,
                "elapsed_seconds": r.elapsed_seconds,
                "violations_count": r.violations_count,
                "violations": r.violations,
            }
            for r in results
        ]

    def watch_directory(self, interval_seconds: int | None = None) -> None:
        interval = interval_seconds or self.container.settings.video_batch.watch_interval
        self.container.watch_directory_uc.execute(
            input_dir=self.input_dir,
            interval_seconds=interval,
            skip_frames=self.skip_frames,
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Leitor e Monitor de Vídeos (.265/.mp4) para Detecção de EPI por Placa e Data"
    )
    parser.add_argument("--dir", "-d", default=None, help="Diretório raiz das gravações")
    parser.add_argument("--file", "-f", default=None, help="Processar um arquivo de vídeo individual")
    parser.add_argument("--watch", "-w", action="store_true", help="Monitoramento contínuo de novas gravações")
    parser.add_argument("--interval", type=int, default=None, help="Intervalo em segundos para checagem")
    parser.add_argument("--skip-frames", type=int, default=None, help="Processar a cada N frames")
    parser.add_argument("--show", action="store_true", help="Exibir janela com o vídeo")
    parser.add_argument("--force", action="store_true", help="Forçar reprocessamento de vídeos já vistos")
    parser.add_argument("--report-csv", default=None, help="Caminho do relatório CSV")
    parser.add_argument("--config", "-c", default=None, help="Arquivo settings.yaml customizado")

    args = parser.parse_args()

    processor = VideoBatchProcessor(
        input_dir=args.dir,
        config_path=args.config,
        skip_frames=args.skip_frames,
        show_window=args.show,
        report_csv=args.report_csv,
        force_reprocess=args.force,
    )

    if args.file:
        run_single_file_command(processor.container, args.file, skip_frames=args.skip_frames)
    elif args.watch:
        run_watch_command(
            processor.container, input_dir=args.dir, interval_seconds=args.interval, skip_frames=args.skip_frames
        )
    else:
        run_batch_command(processor.container, input_dir=args.dir, skip_frames=args.skip_frames, force=args.force)


if __name__ == "__main__":
    main()
