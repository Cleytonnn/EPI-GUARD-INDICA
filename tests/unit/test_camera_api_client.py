"""Testes unitários para o adaptador de câmeras remotas (CameraAPIClient)."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from src.core.domain.detection import Detection
from src.core.services.compliance_service import ComplianceService
from src.infrastructure.cctv.camera_api_client import (
    CameraAPIClient,
    RemoteCameraConfig,
    load_cameras_config,
)


def test_load_cameras_config(tmp_path: Path):
    """Verifica o carregamento de câmeras remotas a partir de YAML."""
    yaml_content = """
cameras:
  - camera_id: "cam_patio_01"
    name: "Pátio Principal"
    url: "http://192.168.1.50/snapshot.jpg"
    auth_type: "basic"
    username: "admin"
    password: "123"
    interval: 1.5
    enabled: true
  - camera_id: "cam_inativa"
    url: "http://192.168.1.51/snapshot.jpg"
    enabled: false
"""
    cfg_file = tmp_path / "cameras.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cameras = load_cameras_config(cfg_file)
    assert len(cameras) == 2
    assert cameras[0].camera_id == "cam_patio_01"
    assert cameras[0].auth_type == "basic"
    assert cameras[0].interval == 1.5
    assert cameras[1].enabled is False


def test_camera_api_client_processes_snapshot():
    """Verifica se o CameraAPIClient processa um snapshot e aciona compliance e storage."""
    cam = RemoteCameraConfig(
        camera_id="cam_01",
        name="Câmera 01",
        url="http://localhost/snap.jpg",
        enabled=True,
    )

    detector = MagicMock()
    detector.detect.return_value = [
        Detection("Person", 0.92, (10, 10, 100, 200), track_id=1),
        Detection("no_helmet", 0.88, (25, 10, 75, 50), track_id=1),
    ]

    alert_storage = MagicMock()
    alert_storage.save_image.return_value = "alertas/cam_01.jpg"
    notifier = MagicMock()

    client = CameraAPIClient(
        cameras=[cam],
        detector=detector,
        compliance_factory=lambda: ComplianceService(
            required_ppe=["helmet"],
            violation_classes=["no_helmet"],
            min_violation_frames=1,
        ),
        alert_storage=alert_storage,
        notifier=notifier,
    )

    dummy_img = np.zeros((120, 160, 3), dtype=np.uint8)
    _, jpg_bytes = cv2.imencode(".jpg", dummy_img)

    mock_resp = MagicMock()
    mock_resp.headers = {"Content-Type": "image/jpeg"}
    mock_resp.content = jpg_bytes.tobytes()
    client._sessions["cam_01"].get = MagicMock(return_value=mock_resp)

    client._process_camera("cam_01")

    assert detector.detect.called
    assert alert_storage.save_image.called
    assert notifier.notify.called
