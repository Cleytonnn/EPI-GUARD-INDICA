"""Configurações tipadas do sistema.

Carrega configurações de settings.yaml e variáveis de ambiente (.env),
provendo validação e acesso tipado em todas as camadas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml
from dotenv import load_dotenv

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

_T = TypeVar("_T")


def _build_dataclass(cls_type: type[_T], raw_data: dict[str, Any] | None) -> _T:
    """Instancia uma dataclass filtrando apenas campos declarados (incluindo default_factory)."""
    data = raw_data if isinstance(raw_data, dict) else {}
    valid_keys = {f.name for f in fields(cls_type)}
    filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}
    return cls_type(**filtered)


@dataclass
class CameraConfig:
    """Configurações de fonte de vídeo e captura em tempo real."""

    source: str | int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    reconnect_delay: int = 5


@dataclass
class DetectionConfig:
    """Configurações do modelo de detecção de EPI."""

    backend: str = "hybrid"  # "hybrid" (Pose + YOLOv8 de EPIs), "pose" (apenas heurística) ou "yolo"
    model_path: str = "yolov8n-pose.pt"
    confidence: float = 0.25
    iou_threshold: float = 0.45
    skip_frames: int = 2
    track: bool = True
    imgsz: int = 640
    # --- Modelo de EPIs treinado (metodologia do TCC) e suavização temporal ---
    ppe_model_path: str = "models/epi_best.pt"
    ppe_confidence: float = 0.25
    ppe_imgsz: int = 640
    crop_padding: float = 0.18  # enquadramento: folga ao redor da pessoa no recorte
    smoothing_alpha: float = 0.35  # EMA da evidência por EPI
    present_threshold: float = 0.55  # histerese: limiar para declarar "presente"
    absent_threshold: float = 0.33  # histerese: limiar para declarar "ausente"
    smoothing_min_obs: int = 3  # observações mínimas antes de decidir


@dataclass
class ComplianceConfig:
    """Configurações das regras de conformidade de EPI."""

    min_violation_frames: int = 5
    required_ppe: list[str] = field(default_factory=lambda: ["helmet", "gloves", "goggles", "vest", "boots"])
    violation_classes: list[str] = field(default_factory=lambda: ["no_helmet", "no_gloves", "no_boots", "no_goggle"])


@dataclass
class AlertConfig:
    """Configurações do sistema de alertas e notificações."""

    save_images: bool = True
    save_dir: str = "alertas"
    send_email: bool = False
    cooldown_seconds: int = 60
    image_quality: int = 85
    email_sender: str | None = None
    email_password: str | None = None
    email_receiver: str | None = None
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587


@dataclass
class DisplayConfig:
    """Configurações da janela de exibição."""

    show_window: bool = True
    window_name: str = "Detecção de EPI - Ao Vivo"
    show_fps: bool = True
    bbox_thickness: int = 2


@dataclass
class VideoBatchConfig:
    """Configurações do processamento de arquivos em lote e watcher."""

    input_dir: str = "gravacoes"
    file_extensions: list[str] = field(default_factory=lambda: [".265", ".h265", ".hevc", ".mp4", ".avi", ".mkv"])
    skip_frames: int = 3
    report_csv: str = "relatorio_infracoes.csv"
    processed_db: str = "videos_processados.json"
    watch_interval: int = 30


@dataclass
class ActiveLearningConfig:
    """Configurações da coleta automática de frames para retreino (Aprendizado Ativo)."""

    enabled: bool = True
    output_dir: str = "datasets/coleta_ativa"
    uncertain_min: float = 0.30
    uncertain_max: float = 0.60
    random_rate: float = 0.02
    min_frame_gap: int = 30
    max_per_video: int = 150
    max_pending: int = 5000
    dedup_threshold: int = 6


@dataclass
class ApiConfig:
    """Configurações da API REST."""

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:8000", "http://127.0.0.1:8000"])
    api_key: str | None = None
    max_upload_mb: int = 10


@dataclass
class AppSettings:
    """Configuração global consolidada da aplicação."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    video_batch: VideoBatchConfig = field(default_factory=VideoBatchConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> AppSettings:
        """Carrega e mescla settings.yaml com variáveis de ambiente."""
        load_dotenv()

        if config_path is None:
            config_path = ROOT_DIR / "config" / "settings.yaml"

        raw_data = {}
        path_obj = Path(config_path)
        if path_obj.exists():
            try:
                with open(path_obj, encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
            except Exception:
                raw_data = {}

        # Mapeamento e instanciação tipada
        cam_data = raw_data.get("camera") or {}
        det_data = raw_data.get("detection") or {}
        comp_data = raw_data.get("compliance") or {}
        alert_data = raw_data.get("alerts") or {}
        disp_data = raw_data.get("display") or {}
        batch_data = raw_data.get("video_batch") or {}
        api_data = dict(raw_data.get("api") or {})
        al_data = raw_data.get("active_learning") or {}

        if env_host := os.getenv("API_HOST"):
            api_data["host"] = env_host
        if env_port := os.getenv("API_PORT"):
            api_data["port"] = int(env_port)
        if env_key := os.getenv("API_KEY"):
            api_data["api_key"] = env_key
        if env_cors := os.getenv("CORS_ORIGINS"):
            api_data["cors_origins"] = [o.strip() for o in env_cors.split(",") if o.strip()]

        alerts_cfg = AlertConfig(
            save_images=alert_data.get("save_images", True),
            save_dir=alert_data.get("save_dir", "alertas"),
            send_email=alert_data.get("send_email", False),
            cooldown_seconds=alert_data.get("cooldown_seconds", 60),
            image_quality=alert_data.get("image_quality", 85),
            email_sender=os.getenv("EMAIL_SENDER"),
            email_password=os.getenv("EMAIL_PASSWORD"),
            email_receiver=os.getenv("EMAIL_RECEIVER"),
            smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
        )

        return cls(
            camera=_build_dataclass(CameraConfig, cam_data),
            detection=_build_dataclass(DetectionConfig, det_data),
            compliance=_build_dataclass(ComplianceConfig, comp_data),
            alerts=alerts_cfg,
            display=_build_dataclass(DisplayConfig, disp_data),
            video_batch=_build_dataclass(VideoBatchConfig, batch_data),
            api=_build_dataclass(ApiConfig, api_data),
            active_learning=_build_dataclass(ActiveLearningConfig, al_data),
        )
