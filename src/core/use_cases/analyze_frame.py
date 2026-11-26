"""Caso de uso: Análise de frame individual (para API REST e integrações sob demanda)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation
from src.core.interfaces.detector import IDetector
from src.core.services.compliance_service import ComplianceService


@dataclass
class FrameAnalysisResult:
    """Resultado da análise de um frame avulso."""

    detections: list[Detection]
    violations: list[Violation]
    is_compliant: bool
    summary: dict[str, int]
    # Infrações visíveis NESTE quadro (ainda não confirmadas pela regra temporal de N quadros)
    frame_violations: list[Detection] = field(default_factory=list)


class AnalyzeFrameUseCase:
    """Processa um frame de imagem sob demanda.

    Mantém um ``ComplianceService`` por ``camera_id`` (quando há ``compliance_factory``),
    para que requisições de câmeras/clientes diferentes não misturem a contagem temporal.
    """

    def __init__(
        self,
        detector: IDetector,
        compliance_service: ComplianceService,
        compliance_factory: Callable[[], ComplianceService] | None = None,
        max_cameras: int = 256,
    ):
        self.detector = detector
        self.compliance_service = compliance_service
        self.compliance_factory = compliance_factory
        self.max_cameras = max(1, max_cameras)
        self._per_camera: OrderedDict[str, ComplianceService] = OrderedDict()
        self._lock = threading.Lock()

    def _service_for(self, camera_id: str | None) -> ComplianceService:
        if self.compliance_factory is None or not camera_id:
            return self.compliance_service
        with self._lock:
            svc = self._per_camera.get(camera_id)
            if svc is None:
                svc = self.compliance_factory()
                self._per_camera[camera_id] = svc
                while len(self._per_camera) > self.max_cameras:  # limita memória (LRU)
                    self._per_camera.popitem(last=False)
            else:
                self._per_camera.move_to_end(camera_id)
            return svc

    def execute(
        self,
        frame: np.ndarray,
        camera_id: str | None = None,
        check_compliance: bool = True,
    ) -> FrameAnalysisResult:
        """Executa a detecção e compliance em um frame."""
        detections = self.detector.detect(frame, skip_frames=1)

        violations: list[Violation] = []
        frame_violations: list[Detection] = []
        if check_compliance:
            meta = None
            if camera_id:
                meta = VehicleMetadata(
                    plate=camera_id,
                    record_date="",
                    video_name="",
                )
            service = self._service_for(camera_id)
            with self._lock:  # o serviço guarda estado entre chamadas
                violations = service.check(detections, metadata=meta)
            frame_violations = [d for d in detections if d.class_name in service.violation_classes]

        summary: dict[str, int] = {}
        for d in detections:
            summary[d.class_name] = summary.get(d.class_name, 0) + 1

        return FrameAnalysisResult(
            detections=detections,
            violations=violations,
            # Antes: só "violations" confirmadas (N quadros seguidos) — uma imagem avulsa sempre saía conforme
            is_compliant=not violations and not frame_violations,
            summary=summary,
            frame_violations=frame_violations,
        )
