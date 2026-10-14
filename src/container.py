"""Container de Inversão de Controle e Injeção de Dependências (IoC/DI).

Centraliza a instanciação e configuração de adaptadores de infraestrutura,
serviços e casos de uso da aplicação.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from src.config.settings import AppSettings
from src.core.services.compliance_service import ComplianceService
from src.core.use_cases.process_video_file import ProcessVideoFileUseCase
from src.core.use_cases.process_batch import ProcessBatchUseCase
from src.core.use_cases.watch_directory import WatchDirectoryUseCase
from src.core.use_cases.analyze_frame import AnalyzeFrameUseCase
from src.core.use_cases.process_stream import ProcessLiveStreamUseCase
from src.infrastructure.ai.yolo_detector import YOLODetector
from src.infrastructure.ai.pose_detector import PoseEPIDetector
from src.infrastructure.storage.file_alert_storage import FileAlertStorage
from src.infrastructure.notifications.email_notifier import EmailNotifier
from src.infrastructure.notifications.composite_notifier import CompositeNotifier
from src.infrastructure.notifications.rate_limiter import RateLimiter
from src.infrastructure.repositories.json_processed_repo import JsonProcessedVideosRepository
from src.infrastructure.repositories.csv_report_repo import CsvViolationReportRepository
from src.infrastructure.video.camera_stream import CameraStream
from src.infrastructure.video.universal_reader import UniversalVideoReader


class Container:
    """Fábrica de serviços e casos de uso da Clean Architecture."""

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or AppSettings.load()

        # 1. Infraestrutura - Detector de IA (Pose ou YOLO BBox)
        backend = getattr(self.settings.detection, "backend", "pose")
        if backend == "pose":
            model_file = self.settings.detection.model_path
            # Se o model_path for o padrão legado de bbox, usa o modelo de pose
            if "best.pt" in model_file:
                model_file = "yolov8n-pose.pt"
            self.detector = PoseEPIDetector(
                model_path=model_file,
                confidence=self.settings.detection.confidence,
                iou_threshold=self.settings.detection.iou_threshold,
                imgsz=self.settings.detection.imgsz,
                track=self.settings.detection.track,
            )
        else:
            self.detector = YOLODetector(
                model_path=self.settings.detection.model_path,
                confidence=self.settings.detection.confidence,
                iou_threshold=self.settings.detection.iou_threshold,
                imgsz=self.settings.detection.imgsz,
                track=self.settings.detection.track,
            )

        # 2. Infraestrutura - Armazenamento de Imagens
        self.alert_storage = FileAlertStorage(
            base_dir=self.settings.alerts.save_dir,
            image_quality=self.settings.alerts.image_quality,
        )

        # 3. Infraestrutura - Notificações (Email + Rate Limiting)
        self.rate_limiter = RateLimiter(
            cooldown_seconds=self.settings.alerts.cooldown_seconds
        )
        self.email_notifier = EmailNotifier(
            sender=self.settings.alerts.email_sender,
            password=self.settings.alerts.email_password,
            receiver=self.settings.alerts.email_receiver,
            smtp_server=self.settings.alerts.smtp_server,
            smtp_port=self.settings.alerts.smtp_port,
            enabled=self.settings.alerts.send_email,
        )
        self.notifier = CompositeNotifier(
            notifiers=[self.email_notifier] if self.settings.alerts.send_email else [],
            rate_limiter=self.rate_limiter,
        )

        # 4. Infraestrutura - Repositórios de Persistência
        self.processed_repo = JsonProcessedVideosRepository(
            db_path=self.settings.video_batch.processed_db
        )
        self.report_repo = CsvViolationReportRepository(
            csv_path=self.settings.video_batch.report_csv
        )

        # 5. Fábrica de Leitores de Vídeo (Universal: .265, .mp4, etc.)
        self.video_reader_factory = lambda path: UniversalVideoReader(path)

        # 6. Serviços de Domínio
        self.compliance_service = ComplianceService(
            required_ppe=self.settings.compliance.required_ppe,
            violation_classes=self.settings.compliance.violation_classes,
            min_violation_frames=self.settings.compliance.min_violation_frames,
        )

        # 7. Casos de Uso
        self.process_video_file_uc = ProcessVideoFileUseCase(
            detector=self.detector,
            compliance_service=self.compliance_service,
            alert_storage=self.alert_storage,
            report_repo=self.report_repo,
            processed_repo=self.processed_repo,
            notifier=self.notifier,
            reader_factory=self.video_reader_factory,
        )

        self.process_batch_uc = ProcessBatchUseCase(
            process_video_uc=self.process_video_file_uc,
            processed_repo=self.processed_repo,
            supported_extensions=self.settings.video_batch.file_extensions,
        )

        self.watch_directory_uc = WatchDirectoryUseCase(
            batch_uc=self.process_batch_uc
        )

        self.analyze_frame_uc = AnalyzeFrameUseCase(
            detector=self.detector,
            compliance_service=self.compliance_service,
        )

        self.process_stream_uc = ProcessLiveStreamUseCase(
            detector=self.detector,
            compliance_service=self.compliance_service,
            alert_storage=self.alert_storage,
            notifier=self.notifier,
        )

    def create_camera_stream(self, source: Optional[int | str] = None) -> CameraStream:
        """Cria uma instância de stream de câmera com a fonte fornecida ou configurada."""
        src = source if source is not None else self.settings.camera.source
        return CameraStream(
            source=src,
            width=self.settings.camera.width,
            height=self.settings.camera.height,
            fps=self.settings.camera.fps,
            reconnect_delay=self.settings.camera.reconnect_delay,
        )
