"""Caso de uso: Análise de frame individual (para API REST e integrações sob demanda)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation
from src.core.interfaces.detector import IDetector
from src.core.services.compliance_service import ComplianceService


@dataclass
class FrameAnalysisResult:
    """Resultado da análise de um frame avulso."""
    detections: List[Detection]
    violations: List[Violation]
    is_compliant: bool
    summary: Dict[str, int]


class AnalyzeFrameUseCase:
    """Processa um frame de imagem sob demanda."""

    def __init__(self, detector: IDetector, compliance_service: ComplianceService):
        self.detector = detector
        self.compliance_service = compliance_service

    def execute(
        self,
        frame: np.ndarray,
        camera_id: Optional[str] = None,
        check_compliance: bool = True,
    ) -> FrameAnalysisResult:
        """Executa a detecção e compliance em um frame."""
        detections = self.detector.detect(frame, skip_frames=1)

        violations: List[Violation] = []
        if check_compliance:
            meta = None
            if camera_id:
                meta = VehicleMetadata(
                    plate=camera_id,
                    record_date="",
                    video_name="",
                )
            violations = self.compliance_service.check(detections, metadata=meta)

        summary: Dict[str, int] = {}
        for d in detections:
            summary[d.class_name] = summary.get(d.class_name, 0) + 1

        return FrameAnalysisResult(
            detections=detections,
            violations=violations,
            is_compliant=len(violations) == 0,
            summary=summary,
        )
