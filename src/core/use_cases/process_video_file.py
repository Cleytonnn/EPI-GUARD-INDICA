"""Caso de uso: Processamento e análise de um arquivo de vídeo individual."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation
from src.core.interfaces.detector import IDetector
from src.core.interfaces.video_reader import IVideoReader
from src.core.interfaces.alert_storage import IAlertStorage
from src.core.interfaces.report_repo import IViolationReportRepository
from src.core.interfaces.processed_repo import IProcessedVideosRepository
from src.core.interfaces.notifier import INotifier
from src.core.interfaces.training_collector import ITrainingSampleCollector
from src.core.services.compliance_service import ComplianceService
from src.presentation.cli.visualizer import draw_detections, draw_vehicle_metadata

logger = logging.getLogger(__name__)


@dataclass
class VideoProcessingResult:
    """Resultado estruturado da execução da análise de um vídeo."""
    success: bool
    video_path: str
    plate: str
    record_date: str
    duration_seconds: float
    elapsed_seconds: float
    violations_count: int
    violations: List[Dict[str, Any]]
    error: Optional[str] = None


class ProcessVideoFileUseCase:
    """Orquestra o ciclo completo de análise de um arquivo de vídeo."""

    def __init__(
        self,
        detector: IDetector,
        compliance_service: ComplianceService,
        alert_storage: IAlertStorage,
        report_repo: IViolationReportRepository,
        processed_repo: IProcessedVideosRepository,
        notifier: Optional[INotifier] = None,
        reader_factory: Optional[Callable[[Path | str], IVideoReader]] = None,
        training_collector: Optional[ITrainingSampleCollector] = None,
    ):
        self.detector = detector
        self.compliance_service = compliance_service
        self.alert_storage = alert_storage
        self.report_repo = report_repo
        self.processed_repo = processed_repo
        self.notifier = notifier
        self.reader_factory = reader_factory
        self.training_collector = training_collector

    def execute(
        self,
        video_path: Path | str,
        root_dir: Optional[Path | str] = None,
        skip_frames: int = 3,
        on_frame_callback: Optional[Callable[[np.ndarray, List[Any], int, int], None]] = None,
    ) -> VideoProcessingResult:
        """Executa a análise frame a frame do vídeo."""
        path_obj = Path(video_path).resolve()
        metadata = VehicleMetadata.from_path(path_obj, root_dir)

        logger.info("=" * 60)
        logger.info(f"🎬 PROCESSANDO VÍDEO: {path_obj.name}")
        logger.info(f"   🚗 Placa Veículo: {metadata.plate}")
        logger.info(f"   📅 Data Gravação: {metadata.record_date}")
        logger.info(f"   📂 Caminho: {path_obj}")
        logger.info("=" * 60)

        # Instanciar reader usando a factory injetada
        if self.reader_factory:
            reader = self.reader_factory(path_obj)
        else:
            from src.infrastructure.video.universal_reader import UniversalVideoReader
            reader = UniversalVideoReader(path_obj)

        if not reader.is_opened():
            logger.error(f"❌ Não foi possível abrir o arquivo de vídeo: {path_obj}")
            return VideoProcessingResult(
                success=False,
                video_path=str(path_obj),
                plate=metadata.plate,
                record_date=metadata.record_date,
                duration_seconds=0.0,
                elapsed_seconds=0.0,
                violations_count=0,
                violations=[],
                error="open_failed",
            )

        fps = reader.fps or 25.0
        total_frames = reader.total_frames
        duration_seconds = total_frames / fps if (fps > 0 and total_frames > 0) else 0.0

        # Resetar estados para o novo vídeo
        self.compliance_service.reset()
        self.detector.reset_tracking()
        if hasattr(self.detector, "is_cabin"):
            self.detector.is_cabin = bool(metadata.is_cabin)
        if self.training_collector:
            self.training_collector.start_session(metadata.video_name)

        frame_idx = 0
        violations_found: List[Dict[str, Any]] = []
        start_time = time.time()
        skip_n = max(1, skip_frames)

        try:
            while True:
                ret, frame = reader.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if frame_idx % skip_n != 0:
                    continue

                video_second = frame_idx / fps
                current_meta = metadata.with_time(video_second, frame_idx)

                # 1. Detecção YOLO
                detections = self.detector.detect(frame, skip_frames=1)

                # 2. Avaliação de Conformidade
                violations = self.compliance_service.check(detections, metadata=current_meta)

                # 2.1 Aprendizado Ativo: guarda frames duvidosos/alertas (frame original, sem anotações)
                if self.training_collector:
                    self.training_collector.consider(frame, detections, violations, current_meta)

                # 3. Tratamento de Violações Confirmadas
                if violations:
                    # Desenhar anotações e tarja de identificação
                    annotated = draw_detections(frame, detections)
                    annotated = draw_vehicle_metadata(
                        annotated,
                        plate=current_meta.plate,
                        record_date=current_meta.record_date,
                        video_name=current_meta.video_name,
                        video_time=current_meta.video_time_str,
                    )

                    for viol in violations:
                        # Salvar imagem
                        saved_path = self.alert_storage.save_image(annotated, viol)

                        # Registrar no relatório CSV
                        self.report_repo.log_violation(viol, image_path=saved_path)

                        # Notificar (ex: email)
                        if self.notifier:
                            self.notifier.notify(viol, image_path=saved_path)

                        violations_found.append({
                            "type": viol.violation_type,
                            "desc": viol.description,
                            "time": current_meta.video_time_str,
                            "second": video_second,
                            "confidence": viol.confidence,
                            "image": saved_path,
                        })

                # Callback visual opcional
                if on_frame_callback:
                    on_frame_callback(frame, detections, frame_idx, total_frames)

        finally:
            reader.release()

        elapsed = time.time() - start_time
        fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0

        logger.info(
            f"✅ Vídeo concluído em {elapsed:.1f}s ({fps_proc:.1f} FPS médio). "
            f"Infrações identificadas: {len(violations_found)}"
        )

        # Registrar no repositório de idempotência
        self.processed_repo.mark_processed(
            video_path=path_obj,
            metadata=metadata,
            violations_count=len(violations_found),
            duration_seconds=duration_seconds,
        )

        return VideoProcessingResult(
            success=True,
            video_path=str(path_obj),
            plate=metadata.plate,
            record_date=metadata.record_date,
            duration_seconds=duration_seconds,
            elapsed_seconds=elapsed,
            violations_count=len(violations_found),
            violations=violations_found,
        )
