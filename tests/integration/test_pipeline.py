"""Testes de integração ponta a ponta da nova arquitetura."""

import numpy as np

from src.config.settings import AppSettings
from src.container import Container
from src.presentation.api.app import create_app


def test_container_initialization_and_settings():
    settings = AppSettings.load()
    container = Container(settings)

    assert container.detector is not None
    assert container.compliance_service is not None
    assert container.alert_storage is not None
    assert container.processed_repo is not None
    assert container.report_repo is not None


def test_analyze_frame_use_case_with_dummy_image():
    settings = AppSettings.load()
    container = Container(settings)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = container.analyze_frame_uc.execute(dummy_frame, camera_id="test_cam")

    assert result is not None
    assert isinstance(result.detections, list)
    assert isinstance(result.violations, list)
    assert result.is_compliant is True


def test_fastapi_app_creation_and_routes():
    settings = AppSettings.load()
    container = Container(settings)
    app = create_app(container)

    paths = list(app.openapi()["paths"].keys())
    assert "/health" in paths
    assert "/analyze/upload" in paths
    assert "/analyze/base64" in paths
    assert "/analyze/annotated" in paths
    assert "/" in paths
