"""Caso de uso: Processamento contínuo de fluxo de câmera ao vivo ou RTSP."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional
import cv2
import numpy as np

from src.core.interfaces.detector import IDetector
from src.core.interfaces.alert_storage import IAlertStorage
from src.core.interfaces.notifier import INotifier
from src.core.interfaces.training_collector import ITrainingSampleCollector
from src.core.services.compliance_service import ComplianceService
from src.infrastructure.video.camera_stream import CameraStream
from src.presentation.cli.visualizer import (
    FPSCounter,
    draw_detections,
    draw_status_bar,
    draw_violation_warning,
)

logger = logging.getLogger(__name__)


class ProcessLiveStreamUseCase:
    """Orquestra a captura ao vivo, detecção, compliance e exibição visual."""

    def __init__(
        self,
        detector: IDetector,
        compliance_service: ComplianceService,
        alert_storage: IAlertStorage,
        notifier: Optional[INotifier] = None,
        training_collector: Optional[ITrainingSampleCollector] = None,
    ):
        self.detector = detector
        self.compliance_service = compliance_service
        self.alert_storage = alert_storage
        self.notifier = notifier
        self.training_collector = training_collector
        self._running = False

    def execute(
        self,
        camera_stream: CameraStream,
        skip_frames: int = 2,
        show_window: bool = True,
        window_name: str = "Detecção de EPI - Ao Vivo",
    ) -> None:
        """Inicia o ciclo de captura e exibição em tempo real."""
        if not camera_stream.start():
            logger.error("❌ Não foi possível iniciar o stream da câmera.")
            return

        self._running = True
        fps_counter = FPSCounter()
        frame_idx = 0
        total_violations_seen = 0
        if self.training_collector:
            self.training_collector.start_session(f"ao_vivo_{int(time.time())}")

        logger.info("=" * 60)
        logger.info("🚀 MONITORAMENTO AO VIVO INICIADO")
        logger.info("   Pressione 'q' na janela ou Ctrl+C no terminal para encerrar.")
        logger.info("=" * 60)

        try:
            while self._running:
                ret, frame = camera_stream.read(timeout=1.0)
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame_idx += 1
                fps = fps_counter.update()

                if frame_idx % max(1, skip_frames) != 0:
                    continue

                # 1. Inferência
                detections = self.detector.detect(frame, skip_frames=1)

                # 2. Avaliação de conformidade
                violations = self.compliance_service.check(detections)

                # 2.1 Aprendizado Ativo
                if self.training_collector:
                    self.training_collector.consider(frame, detections, violations)

                # 3. Tratamento de violações
                if violations:
                    total_violations_seen += len(violations)
                    annotated_alert = draw_detections(frame, detections)
                    for viol in violations:
                        saved_path = self.alert_storage.save_image(annotated_alert, viol)
                        if self.notifier:
                            self.notifier.notify(viol, image_path=saved_path)

                # 4. Renderização visual opcional
                if show_window:
                    display_frame = draw_detections(frame, detections)
                    display_frame = draw_status_bar(
                        display_frame,
                        fps=fps,
                        violations_count=total_violations_seen,
                        alerts_sent=getattr(self.alert_storage, "total_saved", 0),
                    )

                    if violations:
                        display_frame = draw_violation_warning(
                            display_frame, violations[0].description
                        )

                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        logger.info("🛑 Encerramento solicitado pelo operador.")
                        break

        except KeyboardInterrupt:
            logger.info("🛑 Interrupção detectada.")
        finally:
            self._running = False
            camera_stream.stop()
            if show_window:
                cv2.destroyAllWindows()
