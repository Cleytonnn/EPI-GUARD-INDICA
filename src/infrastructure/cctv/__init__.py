"""Adaptadores de integração com sistemas CFTV/VMS externos."""

from src.infrastructure.cctv.camera_api_client import (
    CameraAPIClient,
    RemoteCameraConfig,
    load_cameras_config,
)

__all__ = ["CameraAPIClient", "RemoteCameraConfig", "load_cameras_config"]
