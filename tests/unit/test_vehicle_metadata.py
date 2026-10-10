"""Testes unitários do Value Object VehicleMetadata."""

from pathlib import Path
from src.core.domain.vehicle_metadata import VehicleMetadata


def test_vehicle_metadata_standard_path():
    path = Path("gravacoes/ABC1D23/2026-10-04/camera_01.265")
    meta = VehicleMetadata.from_path(path, root_dir="gravacoes")

    assert meta.plate == "ABC1D23"
    assert meta.record_date == "2026-10-04"
    assert meta.video_name == "camera_01.265"


def test_vehicle_metadata_dvr_filename_date_extraction():
    # Simula arquivo com padrão de DVR UYYYYMMDD-HHMMSS...
    path = Path("gravacoes/BRA2E19/sem_data/U20260803-23492916N300.265")
    meta = VehicleMetadata.from_path(path, root_dir="gravacoes")

    assert meta.plate == "BRA2E19"
    assert meta.record_date == "2026-08-03"


def test_vehicle_metadata_time_formatting():
    meta = VehicleMetadata(plate="TEST123", record_date="2026-10-04", video_name="video.mp4")
    updated = meta.with_time(seconds=125.0, frame_idx=375)

    assert updated.video_second == 125.0
    assert updated.video_time_str == "00:02:05"
    assert updated.frame_index == 375
