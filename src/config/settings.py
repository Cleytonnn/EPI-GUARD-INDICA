"""Configurações tipadas do sistema.

Carrega configurações de settings.yaml e variáveis de ambiente (.env),
provendo validação e acesso tipado em todas as camadas.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def resolve_path(value: str | Path) -> Path:
    """Resolve caminhos relativos a partir da raiz do projeto (e não do diretório de execução)."""
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _build(cls: type, data: dict[str, Any] | None, section: str) -> Any:
    """Instancia uma dataclass a partir de um dicionário, avisando sobre chaves desconhecidas.

    Usa ``dataclasses.fields`` em vez de ``hasattr``: campos com ``default_factory``
    (listas) NÃO existem como atributo de classe, e eram descartados silenciosamente.
    """
    data = data or {}
    if not isinstance(data, dict):
        logger.warning("⚠️ Seção '%s' do settings.yaml não é um mapeamento — usando padrões.", section)
        return cls()
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        logger.warning("⚠️ Chaves desconhecidas em '%s' (ignoradas): %s", section, ", ".join(unknown))
    return cls(**{k: v for k, v in data.items() if k in names})


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("⚠️ Variável %s inválida (%r) — usando %s.", name, raw, default)
        return default


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

    host: str = "127.0.0.1"  # Padrão seguro: só a máquina local. Use 0.0.0.0 explicitamente (ex.: Docker).
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost", "http://127.0.0.1"])
    max_upload_mb: float = 10.0  # Limite de tamanho de imagem enviada para a API
    api_key: str | None = None  # Lido de API_KEY no .env; quando definido, exigido no header X-API-Key


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

        raw_data: dict[str, Any] = {}
        path_obj = Path(config_path)
        if path_obj.exists():
            try:
                with open(path_obj, encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError) as exc:
                # Antes o erro era engolido em silêncio e o sistema rodava com padrões diferentes do esperado
                logger.error("❌ Erro ao ler %s — usando configurações padrão: %s", path_obj, exc)
                raw_data = {}
        else:
            logger.warning("⚠️ Arquivo de configuração não encontrado: %s — usando padrões.", path_obj)

        alert_data = raw_data.get("alerts", {}) or {}

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
            smtp_port=_env_int("SMTP_PORT", 587),
        )

        api_cfg = _build(ApiConfig, raw_data.get("api"), "api")
        api_cfg.api_key = os.getenv("API_KEY") or api_cfg.api_key

        settings = cls(
            camera=_build(CameraConfig, raw_data.get("camera"), "camera"),
            detection=_build(DetectionConfig, raw_data.get("detection"), "detection"),
            compliance=_build(ComplianceConfig, raw_data.get("compliance"), "compliance"),
            alerts=alerts_cfg,
            display=_build(DisplayConfig, raw_data.get("display"), "display"),
            video_batch=_build(VideoBatchConfig, raw_data.get("video_batch"), "video_batch"),
            api=api_cfg,
            active_learning=_build(ActiveLearningConfig, raw_data.get("active_learning"), "active_learning"),
        )
        settings._resolve_paths()
        return settings

    def _resolve_paths(self) -> None:
        """Ancora caminhos de saída na raiz do projeto, independentemente do diretório de execução."""
        self.alerts.save_dir = str(resolve_path(self.alerts.save_dir))
        self.video_batch.report_csv = str(resolve_path(self.video_batch.report_csv))
        self.video_batch.processed_db = str(resolve_path(self.video_batch.processed_db))
        self.active_learning.output_dir = str(resolve_path(self.active_learning.output_dir))
