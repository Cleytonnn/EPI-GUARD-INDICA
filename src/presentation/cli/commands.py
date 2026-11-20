"""Controladores de linha de comando (CLI) da camada de apresentação."""

from __future__ import annotations

import logging
from pathlib import Path

from src.container import Container

logger = logging.getLogger(__name__)


def run_batch_command(
    container: Container,
    input_dir: str | Path | None = None,
    skip_frames: int | None = None,
    force: bool = False,
) -> int:
    """Executa a análise em lote de todos os arquivos de vídeo pendentes."""
    target_dir = input_dir or container.settings.video_batch.input_dir
    skip = skip_frames if skip_frames is not None else container.settings.video_batch.skip_frames

    results = container.process_batch_uc.execute(
        input_dir=target_dir,
        skip_frames=skip,
        force_reprocess=force,
    )
    return 0 if all(r.success for r in results) else 1


def run_watch_command(
    container: Container,
    input_dir: str | Path | None = None,
    interval_seconds: int | None = None,
    skip_frames: int | None = None,
) -> None:
    """Inicia o monitoramento contínuo de novas gravações."""
    target_dir = input_dir or container.settings.video_batch.input_dir
    interval = interval_seconds or container.settings.video_batch.watch_interval
    skip = skip_frames if skip_frames is not None else container.settings.video_batch.skip_frames

    container.watch_directory_uc.execute(
        input_dir=target_dir,
        interval_seconds=interval,
        skip_frames=skip,
    )


def run_single_file_command(
    container: Container,
    file_path: str | Path,
    skip_frames: int | None = None,
) -> int:
    """Analisa um arquivo de vídeo (.265 ou .mp4) individual."""
    path_obj = Path(file_path)
    if not path_obj.exists():
        logger.error(f"❌ Arquivo não encontrado: {path_obj}")
        return 1

    skip = skip_frames if skip_frames is not None else container.settings.video_batch.skip_frames
    result = container.process_video_file_uc.execute(
        video_path=path_obj,
        skip_frames=skip,
    )
    return 0 if result.success else 1


def run_live_stream_command(
    container: Container,
    source: int | str | None = None,
    no_display: bool = False,
    skip_frames: int | None = None,
) -> None:
    """Inicia o monitoramento em tempo real de webcam ou câmera IP."""
    stream = container.create_camera_stream(source=source)
    skip = skip_frames if skip_frames is not None else container.settings.detection.skip_frames
    show = not no_display and container.settings.display.show_window

    container.process_stream_uc.execute(
        camera_stream=stream,
        skip_frames=skip,
        show_window=show,
        window_name=container.settings.display.window_name,
    )


def run_test_email_command(container: Container) -> int:
    """Testa o envio de e-mail SMTP com credenciais configuradas."""
    logger.info("📧 Testando envio de e-mail SMTP...")
    import time

    from src.core.domain.violation import Violation

    dummy_violation = Violation(
        violation_type="no_helmet",
        track_id=1,
        timestamp=time.time(),
        consecutive_frames=5,
        confidence=0.95,
        bbox=(0, 0, 100, 100),
    )
    success = container.email_notifier.notify(dummy_violation)
    if success:
        logger.info("✅ E-mail de teste disparado!")
        return 0
    else:
        logger.error("❌ Falha no disparo do e-mail de teste.")
        return 1
