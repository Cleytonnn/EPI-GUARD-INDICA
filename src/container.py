"""Container de Inversão de Controle e Injeção de Dependências (IoC/DI).

Centraliza a instanciação e configuração de adaptadores de infraestrutura,
serviços e casos de uso da aplicação.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config.settings import ROOT_DIR, AppSettings
from src.core.services.compliance_service import ComplianceService
from src.core.services.ppe_state_smoother import PPEStateSmoother
from src.core.use_cases.analyze_frame import AnalyzeFrameUseCase
from src.core.use_cases.process_batch import ProcessBatchUseCase
from src.core.use_cases.process_stream import ProcessLiveStreamUseCase
from src.core.use_cases.process_video_file import ProcessVideoFileUseCase
from src.core.use_cases.watch_directory import WatchDirectoryUseCase
from src.infrastructure.ai.hybrid_detector import HybridEPIDetector
from src.infrastructure.ai.model_utils import is_git_lfs_pointer, is_usable_weights
from src.infrastructure.ai.pose_detector import PoseEPIDetector
from src.infrastructure.ai.thread_safe_detector import ThreadSafeDetector
from src.infrastructure.ai.yolo_detector import YOLODetector
from src.infrastructure.notifications.composite_notifier import CompositeNotifier
from src.infrastructure.notifications.email_notifier import EmailNotifier
from src.infrastructure.notifications.rate_limiter import RateLimiter
from src.infrastructure.repositories.csv_report_repo import CsvViolationReportRepository
from src.infrastructure.repositories.json_processed_repo import JsonProcessedVideosRepository
from src.infrastructure.storage.file_alert_storage import FileAlertStorage
from src.infrastructure.training.active_learning_collector import ActiveLearningCollector
from src.infrastructure.video.camera_stream import CameraStream
from src.infrastructure.video.universal_reader import UniversalVideoReader

logger = logging.getLogger(__name__)


class Container:
    """Fábrica de serviços e casos de uso da Clean Architecture."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or AppSettings.load()

        # 1. Infraestrutura - Detector de IA (Pose ou YOLO BBox)
        backend = getattr(self.settings.detection, "backend", "pose")
        if backend not in ("pose", "hybrid", "yolo"):
            logger.warning("⚠️ detection.backend '%s' inválido — usando 'hybrid'.", backend)
            backend = "hybrid"
        self.backend = backend
        if backend in ("pose", "hybrid"):
            model_file = self.settings.detection.model_path
            # Se o model_path apontar para um modelo de bbox (não-pose), usa o modelo de pose padrão.
            # (antes: `"best.pt" in model_file`, que também casava com qualquer nome terminado em best.pt)
            if "pose" not in Path(model_file).stem.lower():
                logger.info("ℹ️ model_path '%s' não é um modelo Pose — usando yolov8n-pose.pt.", model_file)
                model_file = "yolov8n-pose.pt"
            pose_detector = PoseEPIDetector(
                model_path=model_file,
                confidence=self.settings.detection.confidence,
                iou_threshold=self.settings.detection.iou_threshold,
                imgsz=self.settings.detection.imgsz,
                track=self.settings.detection.track,
            )
            raw_detector = self._build_hybrid(pose_detector) if backend == "hybrid" else pose_detector
        else:
            yolo_path = Path(self.settings.detection.model_path)
            if not yolo_path.is_absolute() and (ROOT_DIR / yolo_path).exists():
                yolo_path = ROOT_DIR / yolo_path
            if is_git_lfs_pointer(yolo_path):
                raise RuntimeError(
                    f"O modelo '{yolo_path}' é apenas um ponteiro do Git LFS (pesos não baixados). "
                    "Rode 'git lfs install' e 'git lfs pull' na pasta do projeto."
                )
            raw_detector = YOLODetector(
                model_path=yolo_path,
                confidence=self.settings.detection.confidence,
                iou_threshold=self.settings.detection.iou_threshold,
                imgsz=self.settings.detection.imgsz,
                track=self.settings.detection.track,
            )
        # Acesso serializado: API, câmeras HTTP e Desktop compartilham o mesmo detector entre threads
        self.detector = ThreadSafeDetector(raw_detector)

        # 2. Infraestrutura - Armazenamento de Imagens
        self.alert_storage = FileAlertStorage(
            base_dir=self.settings.alerts.save_dir,
            image_quality=self.settings.alerts.image_quality,
        )

        # 3. Infraestrutura - Notificações (Email + Rate Limiting)
        self.rate_limiter = RateLimiter(cooldown_seconds=self.settings.alerts.cooldown_seconds)
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
        self.processed_repo = JsonProcessedVideosRepository(db_path=self.settings.video_batch.processed_db)
        self.report_repo = CsvViolationReportRepository(csv_path=self.settings.video_batch.report_csv)

        # 5. Fábrica de Leitores de Vídeo (Universal: .265, .mp4, etc.)
        self.video_reader_factory = lambda path: UniversalVideoReader(path)

        # 6. Serviços de Domínio
        self.compliance_service = self.create_compliance_service()

        # 6.1 Aprendizado Ativo - coleta de frames para revisão e retreino
        al_cfg = self.settings.active_learning
        self.training_collector = ActiveLearningCollector(
            output_dir=al_cfg.output_dir,
            enabled=al_cfg.enabled,
            uncertain_min=al_cfg.uncertain_min,
            uncertain_max=al_cfg.uncertain_max,
            random_rate=al_cfg.random_rate,
            min_frame_gap=al_cfg.min_frame_gap,
            max_per_video=al_cfg.max_per_video,
            max_pending=al_cfg.max_pending,
            dedup_threshold=al_cfg.dedup_threshold,
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
            training_collector=self.training_collector,
        )

        self.process_batch_uc = ProcessBatchUseCase(
            process_video_uc=self.process_video_file_uc,
            processed_repo=self.processed_repo,
            supported_extensions=self.settings.video_batch.file_extensions,
        )

        self.watch_directory_uc = WatchDirectoryUseCase(batch_uc=self.process_batch_uc)

        self.analyze_frame_uc = AnalyzeFrameUseCase(
            detector=self.detector,
            compliance_service=self.compliance_service,
            compliance_factory=self.create_compliance_service,
        )

        self.process_stream_uc = ProcessLiveStreamUseCase(
            detector=self.detector,
            compliance_service=self.compliance_service,
            alert_storage=self.alert_storage,
            notifier=self.notifier,
            training_collector=self.training_collector,
        )

    def create_compliance_service(self) -> ComplianceService:
        """Cria um ComplianceService novo com as regras do settings (um por câmera/fluxo).

        No modo híbrido o detector já emite ``no_<epi>`` estabilizado; inferir ausência
        de novo geraria alertas duplicados — por isso ``infer_missing_ppe`` depende do backend.
        """
        comp = self.settings.compliance
        return ComplianceService(
            required_ppe=comp.required_ppe,
            violation_classes=comp.violation_classes,
            min_violation_frames=comp.min_violation_frames,
            infer_missing_ppe=(self.backend != "hybrid"),
        )

    def _build_hybrid(self, pose_detector: PoseEPIDetector):
        """Cria o detector híbrido; recua para Pose puro se o modelo de EPIs não existir."""
        det = self.settings.detection
        model_path = Path(det.ppe_model_path)
        if not model_path.is_absolute():
            model_path = ROOT_DIR / model_path
        if not is_usable_weights(model_path):
            if is_git_lfs_pointer(model_path):
                reason = "é apenas um ponteiro do Git LFS (rode 'git lfs install' e 'git lfs pull')"
            else:
                reason = "não foi encontrado"
            logger.warning(
                "⚠️ Modelo de EPIs '%s' %s — usando apenas Pose/heurística. Treine com: python scripts/train_tcc.py",
                model_path,
                reason,
            )
            # Sem o modelo de EPIs a ausência precisa ser inferida pelo ComplianceService
            self.backend = "pose"
            return pose_detector
        smoother = PPEStateSmoother(
            alpha=det.smoothing_alpha,
            present_on=det.present_threshold,
            absent_on=det.absent_threshold,
            min_observations=det.smoothing_min_obs,
        )
        return HybridEPIDetector(
            pose_detector=pose_detector,
            ppe_model_path=model_path,
            ppe_confidence=det.ppe_confidence,
            ppe_imgsz=det.ppe_imgsz,
            crop_padding=det.crop_padding,
            smoother=smoother,
        )

    def create_camera_stream(self, source: int | str | None = None) -> CameraStream:
        """Cria uma instância de stream de câmera com a fonte fornecida ou configurada."""
        src = source if source is not None else self.settings.camera.source
        return CameraStream(
            source=src,
            width=self.settings.camera.width,
            height=self.settings.camera.height,
            fps=self.settings.camera.fps,
            reconnect_delay=self.settings.camera.reconnect_delay,
        )
