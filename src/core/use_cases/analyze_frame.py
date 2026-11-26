"""Caso de uso: Análise de frame individual (para API REST e integrações sob demanda)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

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


class AnalyzeFrameUseCase:
    """Processa um frame de imagem sob demanda."""

    def __init__(
        self,
        detector: IDetector,
        compliance_service: ComplianceService,
        compliance_factory: Callable[[], ComplianceService] | None = None,
    ):
        self.detector = detector
        self.compliance_service = compliance_service
        self.compliance_factory = compliance_factory
        self._compliance_by_camera: dict[str, ComplianceService] = {}
        self._lock = threading.Lock()

    def _get_compliance_service(self, camera_id: str | None) -> ComplianceService:
        if self.compliance_factory is None:
            return self.compliance_service
        key = camera_id or "default"
        service = self._compliance_by_camera.get(key)
        if service is None:
            service = self.compliance_factory()
            self._compliance_by_camera[key] = service
        return service

    def execute(
        self,
        frame: np.ndarray,
        camera_id: str | None = None,
        check_compliance: bool = True,
    ) -> FrameAnalysisResult:
        """Executa a detecção e compliance em um frame."""
        detections = self.detector.detect(frame, skip_frames=1)

        violations: list[Violation] = []
        if check_compliance:
            meta = None
            if camera_id:
                meta = VehicleMetadata(
                    plate=camera_id,
                    record_date="",
                    video_name="",
                )
            with self._lock:
                checker = self._get_compliance_service(camera_id)
                violations = checker.check(detections, metadata=meta)

        summary: dict[str, int] = {}
        for d in detections:
            summary[d.class_name] = summary.get(d.class_name, 0) + 1

        return FrameAnalysisResult(
            detections=detections,
            violations=violations,
            is_compliant=len(violations) == 0,
            summary=summary,
        )
