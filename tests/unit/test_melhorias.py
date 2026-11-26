"""Testes de regressão para os erros corrigidos na revisão (ver MELHORIAS/RELATORIO_MELHORIAS.md)."""

from __future__ import annotations

import base64
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.config.settings import AppSettings
from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation
from src.core.services.compliance_service import ComplianceService
from src.core.use_cases.analyze_frame import AnalyzeFrameUseCase
from src.infrastructure.ai.hybrid_detector import HybridEPIDetector
from src.infrastructure.ai.model_utils import is_git_lfs_pointer, is_usable_weights
from src.infrastructure.ai.thread_safe_detector import ThreadSafeDetector
from src.infrastructure.notifications.rate_limiter import RateLimiter
from src.infrastructure.repositories.csv_report_repo import CsvViolationReportRepository
from src.infrastructure.repositories.json_processed_repo import JsonProcessedVideosRepository
from src.infrastructure.storage.file_alert_storage import FileAlertStorage
from src.infrastructure.training.active_learning_collector import ActiveLearningCollector
from src.infrastructure.video.universal_reader import UniversalVideoReader


# --------------------------------------------------------------- configuração
def test_settings_yaml_list_fields_are_honored(tmp_path):
    """Listas do YAML (required_ppe, violation_classes, extensões, CORS) eram ignoradas."""
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        "compliance:\n  required_ppe: [helmet]\n  violation_classes: [no_helmet]\n"
        "video_batch:\n  file_extensions: ['.mp4']\n"
        "api:\n  cors_origins: ['http://intranet']\n",
        encoding="utf-8",
    )
    s = AppSettings.load(cfg)
    assert s.compliance.required_ppe == ["helmet"]
    assert s.compliance.violation_classes == ["no_helmet"]
    assert s.video_batch.file_extensions == [".mp4"]
    assert s.api.cors_origins == ["http://intranet"]


def test_settings_invalid_yaml_does_not_crash(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("compliance: [isto: nao: e: yaml", encoding="utf-8")
    s = AppSettings.load(cfg)
    assert s.compliance.min_violation_frames == 5  # padrão


def test_settings_paths_are_anchored_on_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = AppSettings.load(tmp_path / "nao_existe.yaml")
    assert s.alerts.save_dir != "alertas"
    assert s.alerts.save_dir.endswith("alertas")


# --------------------------------------------------------------- compliance
def _no_helmet(track_id=None):
    return Detection("no_helmet", 0.9, (10, 10, 50, 50), track_id=track_id)


def test_compliance_empty_frame_resets_counter():
    service = ComplianceService(min_violation_frames=3, infer_missing_ppe=False)
    service.check([_no_helmet(1)])
    service.check([_no_helmet(1)])
    assert service.check([]) == []  # pessoa saiu do quadro
    # Antes a contagem continuava (2 → 3) e confirmava sem 3 quadros consecutivos
    assert service.check([_no_helmet(1)]) == []


def test_compliance_without_tracking_counts_once_per_frame():
    service = ComplianceService(min_violation_frames=3, infer_missing_ppe=False)
    two_people = [_no_helmet(None), Detection("no_helmet", 0.8, (200, 10, 260, 60))]
    assert service.check(two_people) == []
    assert service.check(two_people) == []  # antes: confirmava já aqui (contava 2 por quadro)
    assert len(service.check(two_people)) == 1


def test_compliance_civil_without_track_does_not_silence_workers():
    service = ComplianceService(min_violation_frames=1, infer_missing_ppe=False)
    civil = Detection("civil", 0.9, (300, 0, 400, 200))
    violations = service.check([civil, _no_helmet(None)])
    assert len(violations) == 1


def test_compliance_inference_still_creates_missing_ppe():
    service = ComplianceService(required_ppe=["helmet", "gloves"], min_violation_frames=1)
    person = Detection("Person", 1.0, (0, 0, 100, 200), track_id=7)
    found = {v.violation_type for v in service.check([person])}
    assert found == {"no_helmet", "no_gloves"}


# ------------------------------------------------------------ análise de frame
class _FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, skip_frames=1):
        return list(self.dets)

    def reset_tracking(self):
        pass


def test_analyze_frame_reports_frame_violation_and_isolates_cameras():
    det = _FakeDetector([_no_helmet(1)])

    def factory():
        return ComplianceService(min_violation_frames=2, infer_missing_ppe=False)

    uc = AnalyzeFrameUseCase(det, factory(), compliance_factory=factory)
    frame = np.zeros((10, 10, 3), np.uint8)

    first = uc.execute(frame, camera_id="cam_a")
    assert first.violations == []
    assert first.is_compliant is False  # antes: True (infração visível, mas sem confirmação temporal)
    assert len(first.frame_violations) == 1

    # Outra câmera não herda a contagem da cam_a
    assert uc.execute(frame, camera_id="cam_b").violations == []
    assert len(uc.execute(frame, camera_id="cam_a").violations) == 1


# --------------------------------------------------------------- detector
class _FakePose:
    model_path = "yolov8n-pose.pt"
    is_cabin = False
    confidence = 0.25
    class_names = {0: "Person"}

    def detect(self, frame, skip_frames=1):
        return []

    def reset_tracking(self):
        pass


def test_hybrid_exposes_model_path_and_forwards_is_cabin_through_proxy():
    pose = _FakePose()
    hybrid = HybridEPIDetector(pose, ppe_model=SimpleNamespace(names={0: "helmet"}))
    proxy = ThreadSafeDetector(hybrid)

    assert "yolov8n-pose.pt" in proxy.model_path  # /health dava erro 500 no modo híbrido
    assert hasattr(proxy, "is_cabin")
    proxy.is_cabin = True
    assert pose.is_cabin is True  # antes o modo cabine nunca chegava ao Pose


def test_lfs_pointer_is_detected(tmp_path):
    pointer = tmp_path / "epi_best.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 5477018\n")
    real = tmp_path / "real.pt"
    real.write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    assert is_git_lfs_pointer(pointer)
    assert not is_usable_weights(pointer)
    assert is_usable_weights(real)
    assert not is_usable_weights(tmp_path / "inexistente.pt")


# ----------------------------------------------------------------- vídeo
class _FakeCap:
    def __init__(self, n):
        self.n = n

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        pass


def test_universal_reader_reads_frames_from_vlc_remux(tmp_path):
    reader = UniversalVideoReader.__new__(UniversalVideoReader)
    reader.video_path = "x.265"
    reader._cap = _FakeCap(3)
    reader._container = None
    reader._frame_generator = None
    temp = tmp_path / "vlc_clean_x.mp4"
    temp.write_bytes(b"0" * 10)
    reader._temp_file = temp
    reader.backend = "vlc-remux"

    frames = 0
    while reader.read()[0]:
        frames += 1
    assert frames == 3  # antes: 0 (backend 'vlc-remux' não era lido)
    reader.release()
    assert not temp.exists()  # temporário apagado


# --------------------------------------------------------- persistência
def _violation(meta=None, vtype="no_helmet"):
    return Violation(vtype, 1, 0.0, 5, 0.9, (0, 0, 10, 10), metadata=meta)


def test_alert_storage_blocks_path_traversal(tmp_path):
    storage = FileAlertStorage(base_dir=tmp_path / "alertas")
    meta = VehicleMetadata(plate="../../fora", record_date="2026-01-01", video_name="v.mp4")
    path = storage.save_image(np.zeros((8, 8, 3), np.uint8), _violation(meta))
    assert path is not None
    assert (tmp_path / "alertas").resolve() in __import__("pathlib").Path(path).resolve().parents
    # Segunda evidência com o mesmo nome não sobrescreve a primeira
    path2 = storage.save_image(np.zeros((8, 8, 3), np.uint8), _violation(meta))
    assert path2 != path


def test_csv_report_neutralizes_formulas(tmp_path):
    repo = CsvViolationReportRepository(tmp_path / "r.csv")
    meta = VehicleMetadata(plate="=HYPERLINK(1)", record_date="2026-01-01", video_name="v.mp4")
    repo.log_violation(_violation(meta))
    rows = list(csv.reader(open(tmp_path / "r.csv", encoding="utf-8-sig"), delimiter=";"))
    assert rows[1][1] == "'=HYPERLINK(1)"


def test_json_repo_saves_atomically(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"1")
    repo = JsonProcessedVideosRepository(tmp_path / "db.json")
    meta = VehicleMetadata(plate="ABC", record_date="2026-01-01", video_name="v.mp4")
    repo.mark_processed(video, meta, 0, 1.0)
    assert json.loads((tmp_path / "db.json").read_text(encoding="utf-8"))
    assert not list(tmp_path.glob(".videos_*.tmp"))
    assert JsonProcessedVideosRepository(tmp_path / "db.json").is_processed(video)


def test_active_learning_rejects_unsafe_ids_and_destinations(tmp_path):
    col = ActiveLearningCollector(output_dir=tmp_path / "coleta")
    with pytest.raises(ValueError):
        col.review("../../segredo", "correto")
    frame = np.full((64, 64, 3), 120, np.uint8)
    col.consider(frame, [], [_violation()])
    sample = col.list_samples()[0]["id"]
    col.review(sample, "correto")
    with pytest.raises(ValueError):
        col.export_dataset(dest_dir=tmp_path)  # apagaria a própria coleta


def test_rate_limiter_first_send_always_allowed():
    rl = RateLimiter(cooldown_seconds=60)
    assert rl.can_send("k") is True
    assert rl.can_send("k") is False


# ------------------------------------------------------------------- API
@pytest.fixture
def api_client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.presentation.api.app import create_app

    settings = AppSettings.load()
    settings.api.api_key = "segredo"
    settings.api.max_upload_mb = 0.001  # ~1 KB
    det = ThreadSafeDetector(HybridEPIDetector(_FakePose(), ppe_model=SimpleNamespace(names={0: "helmet"})))
    comp = ComplianceService(infer_missing_ppe=False)
    container = SimpleNamespace(
        settings=settings,
        detector=det,
        alert_storage=SimpleNamespace(total_saved=0),
        email_notifier=SimpleNamespace(total_sent=0),
        analyze_frame_uc=AnalyzeFrameUseCase(det, comp),
    )
    return TestClient(create_app(container))


def test_api_health_works_in_hybrid_mode(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert "yolov8n-pose.pt" in r.json()["model"]


def test_api_requires_key_and_validates_payload(api_client):
    small = base64.b64encode(b"\x89PNG").decode()
    assert api_client.post("/analyze/base64", json={"image_base64": small}).status_code == 401

    headers = {"X-API-Key": "segredo"}
    bad = api_client.post("/analyze/base64", json={"image_base64": "@@@não-é-base64@@@"}, headers=headers)
    assert bad.status_code == 400

    big = base64.b64encode(b"0" * 5000).decode()
    assert api_client.post("/analyze/base64", json={"image_base64": big}, headers=headers).status_code == 413

    files = {"file": ("x.jpg", b"0" * 5000, "image/jpeg")}
    assert api_client.post("/analyze/upload", files=files, headers=headers).status_code == 413
