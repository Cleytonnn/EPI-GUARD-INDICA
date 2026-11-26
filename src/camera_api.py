"""Entrypoint CLI para monitoramento periódico de câmeras CFTV via API HTTP.

Uso:
    python -m src.camera_api
    python -m src.camera_api --config config/cameras.yaml
"""

from __future__ import annotations

import argparse
import logging

from src.config.settings import AppSettings
from src.container import Container
from src.infrastructure.cctv.camera_api_client import (
    CameraAPIClient,
    load_cameras_config,
)
from src.infrastructure.cctv.camera_api_client import (
    RemoteCameraConfig as CameraConfig,
)

logger = logging.getLogger(__name__)

__all__ = ["CameraAPIClient", "CameraConfig", "load_cameras_config", "main"]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Monitor de EPI via API de Câmeras (Clean Architecture)")
    parser.add_argument("--config", "-c", default=None, help="Arquivo de configuração de câmeras (cameras.yaml)")
    parser.add_argument("--settings", "-s", default=None, help="Arquivo settings.yaml alternativo")
    args = parser.parse_args()

    settings = AppSettings.load(args.settings)
    container = Container(settings)

    cameras = load_cameras_config(args.config)
    if not cameras:
        logger.error("❌ Nenhuma câmera configurada. Encerrando.")
        return

    client = CameraAPIClient(
        cameras=cameras,
        detector=container.detector,
        compliance_factory=container.create_compliance_service,
        alert_storage=container.alert_storage,
        notifier=container.notifier,
    )
    client.start()


if __name__ == "__main__":
    main()
