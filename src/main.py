"""Pipeline principal do Sistema de Detecção de EPI em Tempo Real.

Ponto de entrada unificado que orquestra a captura de vídeo ao vivo,
varredura em lote de gravações e despacho de alertas.
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.container import Container
from src.presentation.cli.commands import (
    run_batch_command,
    run_live_stream_command,
    run_test_email_command,
    run_watch_command,
)

logger = logging.getLogger(__name__)


class PPEMonitor:
    """Fachada retrocompatível para monitoramento ao vivo."""

    def __init__(self, config_path: str | None = None):
        from src.config.settings import AppSettings

        settings = AppSettings.load(config_path)
        self.container = Container(settings)

        # Aliases retrocompatíveis
        self.detector = self.container.detector
        self.alerter = self.container.alert_storage
        self.compliance = self.container.compliance_service

    def run(self, source: int | str | None = None, no_display: bool = False) -> None:
        run_live_stream_command(self.container, source=source, no_display=no_display)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de EPI em Tempo Real (Clean Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", "-s", default=None, help="Fonte de vídeo (0=webcam, 'video.mp4', 'rtsp://...')")
    parser.add_argument("--batch-dir", default=None, help="Processar pasta de gravações (Placa > Data > *.265/.mp4)")
    parser.add_argument("--watch", action="store_true", help="Monitorar continuamente a pasta por novas gravações")
    parser.add_argument("--skip-frames", type=int, default=None, help="Processar a cada N frames")
    parser.add_argument("--confidence", type=float, default=None, help="Limiar de confiança (0.0 a 1.0)")
    parser.add_argument("--no-display", action="store_true", help="Executar sem janela visual")
    parser.add_argument("--test-email", action="store_true", help="Enviar e-mail de teste e sair")
    parser.add_argument("--config", "-c", default=None, help="Arquivo de configuração YAML alternativo")

    args = parser.parse_args()

    from src.config.settings import AppSettings

    settings = AppSettings.load(args.config)
    if args.confidence is not None:
        settings.detection.confidence = args.confidence
    if args.skip_frames is not None:
        settings.detection.skip_frames = args.skip_frames
        settings.video_batch.skip_frames = args.skip_frames

    container = Container(settings)

    if args.test_email:
        sys.exit(run_test_email_command(container))

    if args.batch_dir:
        if args.watch:
            run_watch_command(container, input_dir=args.batch_dir, skip_frames=args.skip_frames)
        else:
            sys.exit(run_batch_command(container, input_dir=args.batch_dir, skip_frames=args.skip_frames))
        return

    # Executar monitoramento ao vivo
    src = args.source
    if src is not None:
        try:
            src = int(src)
        except ValueError:
            pass

    run_live_stream_command(container, source=src, no_display=args.no_display, skip_frames=args.skip_frames)


if __name__ == "__main__":
    main()
