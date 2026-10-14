"""Configurações tipadas do sistema.

Carrega configurações de settings.yaml e variáveis de ambiente (.env),
provendo validação e acesso tipado em todas as camadas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml
from dotenv import load_dotenv

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


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
    backend: str = "pose"  # "pose" (YOLO-Pose anatômico) ou "yolo" (bounding box padrão)
    model_path: str = "yolov8n-pose.pt"
    confidence: float = 0.25
    iou_threshold: float = 0.45
    skip_frames: int = 2
    track: bool = True
    imgsz: int = 640


@dataclass
class ComplianceConfig:
    """Configurações das regras de conformidade de EPI."""
    min_violation_frames: int = 5
    required_ppe: List[str] = field(default_factory=lambda: [
        "helmet", "gloves", "goggles", "vest", "boots"
    ])
    violation_classes: List[str] = field(default_factory=lambda: [
        "no_helmet", "no_gloves", "no_boots", "no_goggle"
    ])


@dataclass
class AlertConfig:
    """Configurações do sistema de alertas e notificações."""
    save_images: bool = True
    save_dir: str = "alertas"
    send_email: bool = False
    cooldown_seconds: int = 60
    image_quality: int = 85
    email_sender: Optional[str] = None
    email_password: Optional[str] = None
    email_receiver: Optional[str] = None
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
    file_extensions: List[str] = field(default_factory=lambda: [
        ".265", ".h265", ".hevc", ".mp4", ".avi", ".mkv"
    ])
    skip_frames: int = 3
    report_csv: str = "relatorio_infracoes.csv"
    processed_db: str = "videos_processados.json"
    watch_interval: int = 30


@dataclass
class ApiConfig:
    """Configurações da API REST."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


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

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> AppSettings:
        """Carrega e mescla settings.yaml com variáveis de ambiente."""
        load_dotenv()

        if config_path is None:
            config_path = ROOT_DIR / "config" / "settings.yaml"

        raw_data = {}
        path_obj = Path(config_path)
        if path_obj.exists():
            try:
                with open(path_obj, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
            except Exception:
                raw_data = {}

        # Mapeamento e instanciação tipada
        cam_data = raw_data.get("camera", {})
        det_data = raw_data.get("detection", {})
        comp_data = raw_data.get("compliance", {})
        alert_data = raw_data.get("alerts", {})
        disp_data = raw_data.get("display", {})
        batch_data = raw_data.get("video_batch", {})
        api_data = raw_data.get("api", {})

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
            camera=CameraConfig(**{k: v for k, v in cam_data.items() if hasattr(CameraConfig, k)}),
            detection=DetectionConfig(**{k: v for k, v in det_data.items() if hasattr(DetectionConfig, k)}),
            compliance=ComplianceConfig(**{k: v for k, v in comp_data.items() if hasattr(ComplianceConfig, k)}),
            alerts=alerts_cfg,
            display=DisplayConfig(**{k: v for k, v in disp_data.items() if hasattr(DisplayConfig, k)}),
            video_batch=VideoBatchConfig(**{k: v for k, v in batch_data.items() if hasattr(VideoBatchConfig, k)}),
            api=ApiConfig(**{k: v for k, v in api_data.items() if hasattr(ApiConfig, k)}),
        )
