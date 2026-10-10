"""Schemas e DTOs Pydantic para a API REST."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    """Schema de detecção de EPI."""
    class_name: str
    confidence: float
    bbox: List[int]
    is_violation: bool
    label_pt: str
    track_id: Optional[int] = None


class ViolationItem(BaseModel):
    """Schema de violação confirmada."""
    violation_type: str
    description: str
    confidence: float
    consecutive_frames: int
    track_id: Optional[int] = None


class FrameAnalysisResponse(BaseModel):
    """Resposta consolidada de análise de frame."""
    is_compliant: bool
    total_detections: int
    total_violations: int
    detections: List[DetectionItem]
    violations: List[ViolationItem]
    summary: Dict[str, int]


class Base64AnalyzeRequest(BaseModel):
    """Payload de imagem em Base64."""
    image_base64: str = Field(..., description="String Base64 da imagem (com ou sem prefixo data:image/jpeg;base64)")
    camera_id: Optional[str] = Field("cam_default", description="Identificador único da câmera ou veículo")
    check_compliance: bool = Field(True, description="Se True, aplica motor de conformidade temporal")


class SystemHealthResponse(BaseModel):
    """Status operacional do sistema."""
    status: str
    model: str
    device: str
    classes_count: int
