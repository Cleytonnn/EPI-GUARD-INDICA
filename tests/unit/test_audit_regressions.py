"""Testes de regressão para os apontamentos críticos e de segurança da auditoria."""

from __future__ import annotations

import base64
import concurrent.futures
import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.config.settings import ROOT_DIR, AppSettings
from src.container import Container
from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation
from src.core.services.compliance_service import ComplianceService
from src.core.services.ppe_state_smoother import PPEStateSmoother
from src.core.use_cases.analyze_frame import AnalyzeFrameUseCase
from src.infrastructure.ai.hybrid_detector import HybridEPIDetector
from src.infrastructure.ai.pose_detector import PoseEPIDetector, is_git_lfs_pointer
from src.infrastructure.cctv.camera_api_client import CameraAPIClient, RemoteCameraConfig, load_cameras_config
from src.infrastructure.repositories.csv_report_repo import CsvViolationReportRepository
from src.infrastructure.repositories.json_processed_repo import JsonProcessedVideosRepository
from src.infrastructure.storage.file_alert_storage import FileAlertStorage
from src.infrastructure.video.universal_reader import UniversalVideoReader
from src.presentation.api.app import create_app


# ---------------------------------------------------------------------------
# 1. settings.yaml: listas não descartadas + defaults seguros da API
# ---------------------------------------------------------------------------
def test_settings_loads_yaml_lists_and_secure_api_defaults(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)

    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        """
compliance:
  min_violation_frames: 7
  required_ppe:
    - helmet
    - uniform_rf
  violation_classes:
    - no_helmet
    - no_uniform_rf
video_batch:
  file_extensions:
    - ".mp4"
    - ".265"
api:
  cors_origins:
    - "https://painel.empresa.local"
  max_upload_mb: 5
""",
        encoding="utf-8",
    )

    settings = AppSettings.load(cfg)
    assert settings.compliance.required_ppe == ["helmet", "uniform_rf"]
    assert settings.compliance.violation_classes == ["no_helmet", "no_uniform_rf"]
    assert settings.video_batch.file_extensions == [".mp4", ".265"]
    assert settings.api.cors_origins == ["https://painel.empresa.local"]
    assert settings.api.host == "127.0.0.1"
    assert settings.api.max_upload_mb == 5
    assert settings.api.api_key is None


def test_settings_default_api_cors_is_restricted(tmp_path: Path):
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("{}", encoding="utf-8")
    settings = AppSettings.load(cfg)
    assert settings.api.host == "127.0.0.1"
    assert "*" not in settings.api.cors_origins


# ---------------------------------------------------------------------------
# 2. Ponteiro Git LFS em modelos .pt + fallback seguro para Pose
# ---------------------------------------------------------------------------
def test_git_lfs_pointer_detection_and_container_fallback(tmp_path: Path, caplog):
    lfs_pt = tmp_path / "epi_best.pt"
    lfs_pt.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        "size 22506858\n",
        encoding="utf-8",
    )
    assert is_git_lfs_pointer(lfs_pt) is True

    settings = AppSettings()
    settings.detection.backend = "hybrid"
    settings.detection.ppe_model_path = str(lfs_pt)

    with caplog.at_level("WARNING"):
        container = Container(settings)

    assert isinstance(container.detector, PoseEPIDetector)
    assert container.compliance_service.infer_missing_ppe is True
    assert any("Git LFS" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 3. UniversalVideoReader backend vlc-remux lê frames e remove temporário único
# ---------------------------------------------------------------------------
def test_universal_reader_vlc_remux_reads_frames_and_cleans_temp(tmp_path: Path):
    raw_265 = tmp_path / "sample.265"
    raw_265.write_bytes(b"\x00\x00\x00\x01\x40\x01" + b"\xaa" * 128)

    remuxed_mp4 = tmp_path / "remuxed_unique.mp4"
    writer = cv2.VideoWriter(str(remuxed_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(4):
        frame = np.full((48, 64, 3), i * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    def fake_remux(inp, out):
        Path(out).write_bytes(remuxed_mp4.read_bytes())
        return Path(out)

    with (
        patch.object(UniversalVideoReader, "_try_opencv", return_value=False),
        patch("src.infrastructure.video.vlc_service.VLCService.is_available", return_value=True),
        patch("src.infrastructure.video.vlc_service.VLCService.remux_to_clean_mp4", side_effect=fake_remux),
    ):
        reader = UniversalVideoReader(raw_265)
        try:
            assert reader.is_opened() is True
            assert reader.backend == "vlc-remux"
            temp_path = reader._temp_remux_path
            assert temp_path is not None and temp_path.exists()
            assert temp_path.name != f"vlc_clean_{raw_265.stem}.mp4"

            frames_read = 0
            while True:
                ok, frame = reader.read()
                if not ok or frame is None:
                    break
                frames_read += 1
            assert frames_read > 0
        finally:
            reader.release()

    assert not temp_path.exists()


# ---------------------------------------------------------------------------
# 4. ComplianceService: frame vazio, sem dupla contagem, civil sem track
# ---------------------------------------------------------------------------
def test_compliance_empty_frame_resets_counters():
    service = ComplianceService(min_violation_frames=2, infer_missing_ppe=False)
    det = Detection("no_helmet", 0.9, (10, 10, 50, 50), track_id=1)

    assert service.check([det]) == []
    # Quadro vazio deve zerar o contador acumulado
    assert service.check([]) == []
    # Novo frame com detecção deve reiniciar a contagem em 1 (não confirmar no 1º frame)
    assert service.check([det]) == []
    assert len(service.check([det])) == 1


def test_compliance_no_double_counting_without_tracking_or_two_hands():
    service = ComplianceService(min_violation_frames=2, infer_missing_ppe=False)
    # Duas detecções no mesmo frame sem tracking (track_id=None e track_id=-1)
    d1 = Detection("no_helmet", 0.85, (10, 10, 50, 50), track_id=None)
    d2 = Detection("no_helmet", 0.90, (100, 10, 150, 50), track_id=None)

    assert service.check([d1, d2]) == []
    v = service.check([d1, d2])
    assert len(v) == 1
    assert v[0].consecutive_frames == 2

    # Duas mãos sem luvas do mesmo trabalhador não devem contar 2 frames em 1 frame
    service.reset()
    g1 = Detection("no_gloves", 0.80, (10, 100, 30, 130), track_id=-1)
    g2 = Detection("no_gloves", 0.88, (70, 100, 90, 130), track_id=-1)
    assert service.check([g1, g2]) == []
    v2 = service.check([g1, g2])
    assert len(v2) == 1
    assert v2[0].consecutive_frames == 2


def test_compliance_untracked_civil_does_not_silence_untracked_worker():
    service = ComplianceService(min_violation_frames=1, infer_missing_ppe=False, only_company_workers=True)
    civil = Detection("civil", 0.9, (10, 10, 80, 200), track_id=-1)
    worker_violation = Detection("no_helmet", 0.9, (200, 10, 260, 80), track_id=-1)

    violations = service.check([civil, worker_violation])
    assert len(violations) == 1
    assert violations[0].violation_type == "no_helmet"

    # Também com track_id=None
    service.reset()
    civil_none = Detection("civil", 0.9, (10, 10, 80, 200), track_id=None)
    worker_none = Detection("no_helmet", 0.9, (200, 10, 260, 80), track_id=None)
    violations_none = service.check([civil_none, worker_none])
    assert len(violations_none) == 1


# ---------------------------------------------------------------------------
# 5. HybridEPIDetector: is_cabin, model_path e thread-safety
# ---------------------------------------------------------------------------
class _StubPose:
    model_path = "yolov8n-pose.pt"
    confidence = 0.25
    class_names = {0: "Person"}
    is_cabin = False

    def detect(self, frame, skip_frames=1):
        return [Detection("Person", 0.9, (50, 50, 150, 250), track_id=1)]

    def reset_tracking(self):
        pass


class _StubPPEModel:
    names = {0: "helmet"}

    def predict(self, source, **kwargs):
        class _R:
            boxes = None

        return [_R() for _ in source]


def test_hybrid_detector_exposes_is_cabin_model_path_and_is_thread_safe():
    pose = _StubPose()
    det = HybridEPIDetector(
        pose_detector=pose,
        ppe_model_path="models/epi_best.pt",
        ppe_model=_StubPPEModel(),
        smoother=PPEStateSmoother(min_observations=1),
    )
    assert "epi_best.pt" in det.model_path
    assert det.is_cabin is False
    det.is_cabin = True
    assert pose.is_cabin is True
    assert det.is_cabin is True

    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(det.detect, frame) for _ in range(12)]
        results = [f.result() for f in futures]
    assert all(isinstance(r, list) for r in results)


# ---------------------------------------------------------------------------
# 6. API REST: /health com Hybrid, API_KEY, limite de upload, base64, per-camera
# ---------------------------------------------------------------------------
def _make_jpeg_bytes() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _make_test_container(api_key: str | None = None, max_upload_mb: int = 1):
    settings = AppSettings()
    settings.api.api_key = api_key
    settings.api.max_upload_mb = max_upload_mb
    settings.compliance.min_violation_frames = 2
    settings.compliance.violation_classes = ["no_helmet"]

    fake_detector = HybridEPIDetector(
        pose_detector=_StubPose(),
        ppe_model_path="models/epi_best.pt",
        ppe_model=_StubPPEModel(),
    )
    fake_detector.detect = MagicMock(
        return_value=[
            Detection("Person", 0.9, (10, 10, 80, 90), track_id=1),
            Detection("no_helmet", 0.9, (20, 10, 60, 40), track_id=1),
        ]
    )

    container = MagicMock(spec=Container)
    container.settings = settings
    container.detector = fake_detector
    container.alert_storage = MagicMock(total_saved=0)
    container.email_notifier = MagicMock(total_sent=0)
    container.create_compliance_service = lambda: ComplianceService(
        required_ppe=["helmet"],
        violation_classes=["no_helmet"],
        min_violation_frames=2,
        infer_missing_ppe=False,
    )
    container.compliance_service = container.create_compliance_service()
    container.analyze_frame_uc = AnalyzeFrameUseCase(
        detector=fake_detector,
        compliance_service=container.compliance_service,
        compliance_factory=container.create_compliance_service,
    )
    return container


def test_api_health_with_hybrid_and_optional_api_key():
    container = _make_test_container(api_key="segredo-123")
    app = create_app(container)
    client = TestClient(app)

    # /health funciona mesmo com HybridEPIDetector e sem API Key
    health = client.get("/health")
    assert health.status_code == 200
    assert "epi_best.pt" in health.json()["model"]

    # Endpoints protegidos exigem chave quando API_KEY está configurada
    b64 = base64.b64encode(_make_jpeg_bytes()).decode("ascii")
    assert client.post("/analyze/base64", json={"image_base64": b64}).status_code == 401
    assert client.get("/stats").status_code == 401

    ok_resp = client.post(
        "/analyze/base64",
        json={"image_base64": b64, "camera_id": "cam_1"},
        headers={"X-API-Key": "segredo-123"},
    )
    assert ok_resp.status_code == 200

    bearer_resp = client.get("/stats", headers={"Authorization": "Bearer segredo-123"})
    assert bearer_resp.status_code == 200


def test_api_validates_base64_upload_limit_and_per_camera_isolation():
    container = _make_test_container(api_key=None, max_upload_mb=1)
    app = create_app(container)
    client = TestClient(app)

    # Base64 inválido (caracteres fora do alfabeto) -> 400
    assert client.post("/analyze/base64", json={"image_base64": "!!!not_base64!!!"}).status_code == 400
    # Prefixo data URI inválido -> 400
    assert client.post("/analyze/base64", json={"image_base64": "data:text/plain;base64,AAAA"}).status_code == 400

    # Upload maior que o limite (1 MB) -> 413
    oversized = b"x" * (1024 * 1024 + 10)
    up_resp = client.post("/analyze/upload", files={"file": ("big.jpg", oversized, "image/jpeg")})
    assert up_resp.status_code == 413

    # Isolamento de ComplianceService por câmera:
    # cam_a frame 1 -> não confirma (min=2); cam_b frame 1 -> não mistura com cam_a;
    # cam_a frame 2 -> confirma apenas em cam_a!
    b64 = base64.b64encode(_make_jpeg_bytes()).decode("ascii")
    r_a1 = client.post("/analyze/base64", json={"image_base64": b64, "camera_id": "cam_a"}).json()
    r_b1 = client.post("/analyze/base64", json={"image_base64": b64, "camera_id": "cam_b"}).json()
    r_a2 = client.post("/analyze/base64", json={"image_base64": b64, "camera_id": "cam_a"}).json()

    assert r_a1["total_violations"] == 0
    assert r_b1["total_violations"] == 0
    assert r_a2["total_violations"] == 1


# ---------------------------------------------------------------------------
# 7. Storage/Repositórios: sanitização de caminho, CSV sem fórmulas, JSON atômico
# ---------------------------------------------------------------------------
def test_file_alert_storage_sanitizes_traversal_paths(tmp_path: Path):
    base_dir = tmp_path / "alertas"
    storage = FileAlertStorage(base_dir=base_dir)
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    meta = VehicleMetadata(
        plate="../../etc",
        record_date="..\\..\\Windows",
        video_name="../segredo.mp4",
        video_time_str="00:01:02",
    )
    viol = Violation(
        violation_type="no_helmet",
        track_id=1,
        timestamp=0.0,
        consecutive_frames=3,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        metadata=meta,
    )

    saved = storage.save_image(frame, viol)
    assert saved is not None
    saved_path = Path(saved).resolve()
    assert saved_path.exists()
    assert saved_path.is_relative_to(base_dir.resolve())


def test_csv_report_neutralizes_formula_injection(tmp_path: Path):
    csv_path = tmp_path / "relatorio.csv"
    repo = CsvViolationReportRepository(csv_path=csv_path)
    meta = VehicleMetadata(
        plate="=CMD|' /C calc'!A0",
        record_date="+2026-11-26",
        video_name="@evil.mp4",
        video_time_str="00:00:01",
    )
    viol = Violation(
        violation_type="no_helmet",
        track_id=1,
        timestamp=0.0,
        consecutive_frames=3,
        confidence=0.9,
        bbox=(0, 0, 10, 10),
        metadata=meta,
    )

    repo.log_violation(viol, image_path="-alert.jpg")

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))

    data_row = rows[1]
    assert data_row[1].startswith("'=CMD")
    assert data_row[2].startswith("'+2026")
    assert data_row[3].startswith("'@evil")
    assert data_row[11].startswith("'-alert.jpg")


def test_json_processed_repo_writes_atomically(tmp_path: Path):
    db_path = tmp_path / "videos_processados.json"
    video = tmp_path / "vid.mp4"
    video.write_bytes(b"1234")

    repo = JsonProcessedVideosRepository(db_path=db_path)
    meta = VehicleMetadata(plate="ABC1D23", record_date="2026-11-26", video_name="vid.mp4")
    repo.mark_processed(video, meta, violations_count=2, duration_seconds=12.5)

    loaded = json.loads(db_path.read_text(encoding="utf-8"))
    assert len(loaded) == 1
    assert list(tmp_path.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# 8. cameras.yaml com ${VARIAVEIS} e exemplos desabilitados
# ---------------------------------------------------------------------------
def test_cameras_yaml_examples_disabled_and_env_expansion(tmp_path: Path, monkeypatch):
    default_cams = load_cameras_config(ROOT_DIR / "config" / "cameras.yaml")
    assert len(default_cams) > 0
    assert all(c.enabled is False for c in default_cams)

    monkeypatch.setenv("TEST_CAM_TOKEN", "tok-secreto-999")
    monkeypatch.setenv("TEST_CAM_USER", "operador")
    custom_yaml = tmp_path / "cameras.yaml"
    custom_yaml.write_text(
        """
cameras:
  - camera_id: "cam_env"
    name: "Env Cam"
    url: "${TEST_CAM_URL:-http://127.0.0.1:9000/snap}"
    auth_type: "bearer"
    token: "${TEST_CAM_TOKEN}"
    username: "${TEST_CAM_USER}"
    enabled: true
""",
        encoding="utf-8",
    )
    cams = load_cameras_config(custom_yaml)
    assert len(cams) == 1
    assert cams[0].url == "http://127.0.0.1:9000/snap"
    assert cams[0].token == "tok-secreto-999"
    assert cams[0].username == "operador"


def test_api_cors_restricts_unknown_origin_and_annotated_respects_auth():
    container = _make_test_container(api_key="key-xyz", max_upload_mb=1)
    app = create_app(container)
    client = TestClient(app)

    cors_resp = client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert cors_resp.headers.get("access-control-allow-origin") != "*"
    assert cors_resp.headers.get("access-control-allow-origin") != "https://evil.example.com"

    jpeg = _make_jpeg_bytes()
    assert client.post("/analyze/annotated", files={"file": ("f.jpg", jpeg, "image/jpeg")}).status_code == 401
    ok_ann = client.post(
        "/analyze/annotated",
        files={"file": ("f.jpg", jpeg, "image/jpeg")},
        headers={"X-API-Key": "key-xyz"},
    )
    assert ok_ann.status_code == 200
    assert ok_ann.headers["content-type"] == "image/jpeg"


def test_camera_api_client_isolates_compliance_per_camera():
    cams = [
        RemoteCameraConfig(camera_id="c1", name="Cam 1", url="http://localhost/1.jpg", enabled=True),
        RemoteCameraConfig(camera_id="c2", name="Cam 2", url="http://localhost/2.jpg", enabled=True),
    ]
    detector = MagicMock()
    detector.detect.return_value = [
        Detection("Person", 0.9, (10, 10, 80, 90), track_id=1),
        Detection("no_helmet", 0.9, (20, 10, 60, 40), track_id=1),
    ]
    storage = MagicMock()
    storage.save_image.return_value = "alertas/c.jpg"

    client = CameraAPIClient(
        cameras=cams,
        detector=detector,
        compliance_factory=lambda: ComplianceService(
            required_ppe=["helmet"],
            violation_classes=["no_helmet"],
            min_violation_frames=2,
            infer_missing_ppe=False,
        ),
        alert_storage=storage,
    )

    jpeg = _make_jpeg_bytes()
    for cid in ("c1", "c2"):
        resp = MagicMock()
        resp.headers = {"Content-Type": "image/jpeg"}
        resp.content = jpeg
        client._sessions[cid].get = MagicMock(return_value=resp)

    client._process_camera("c1")
    client._process_camera("c2")
    assert storage.save_image.call_count == 0
    client._process_camera("c1")
    assert storage.save_image.call_count == 1
