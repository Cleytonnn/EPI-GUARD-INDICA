"""Rotas e controladores REST da aplicação."""

from __future__ import annotations

import base64
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from src.container import Container
from src.core.use_cases.analyze_frame import FrameAnalysisResult
from src.presentation.api.schemas import (
    Base64AnalyzeRequest,
    DetectionItem,
    FrameAnalysisResponse,
    SystemHealthResponse,
    ViolationItem,
)
from src.presentation.cli.visualizer import draw_detections

router = APIRouter()


def get_container() -> Container:
    """Dependency provider do container de injeção."""
    from src.presentation.api.app import app_container
    return app_container


def _decode_bytes_to_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem fornecida.")
    return frame


def _to_response_dto(result: FrameAnalysisResult) -> FrameAnalysisResponse:
    det_items = [
        DetectionItem(
            class_name=d.class_name,
            confidence=d.confidence,
            bbox=list(d.bbox),
            is_violation=d.is_violation,
            label_pt=d.label_pt,
            track_id=d.track_id,
        )
        for d in result.detections
    ]
    viol_items = [
        ViolationItem(
            violation_type=v.violation_type,
            description=v.description,
            confidence=v.confidence,
            consecutive_frames=v.consecutive_frames,
            track_id=v.track_id,
        )
        for v in result.violations
    ]
    return FrameAnalysisResponse(
        is_compliant=result.is_compliant,
        total_detections=len(det_items),
        total_violations=len(viol_items),
        detections=det_items,
        violations=viol_items,
        summary=result.summary,
    )


@router.get("/health", response_model=SystemHealthResponse, tags=["Monitoramento"])
def health_check(container: Container = Depends(get_container)):
    """Verifica saúde do servidor e dispositivo de IA."""
    det = container.detector
    return SystemHealthResponse(
        status="healthy",
        model=det.model_path,
        device=det.device,
        classes_count=len(det.class_names),
    )


@router.get("/stats", tags=["Monitoramento"])
def get_system_stats(container: Container = Depends(get_container)):
    """Retorna estatísticas operacionais de alertas e processamento."""
    return {
        "imagens_salvas": getattr(container.alert_storage, "total_saved", 0),
        "emails_enviados": getattr(container.email_notifier, "total_sent", 0),
    }



@router.post("/analyze/upload", response_model=FrameAnalysisResponse, tags=["Inferência"])
async def analyze_file_upload(
    file: UploadFile = File(..., description="Arquivo de imagem (JPEG/PNG)"),
    camera_id: str = Form("cam_upload", description="Identificador da câmera"),
    check_compliance: bool = Form(True, description="Verificar conformidade temporal"),
    container: Container = Depends(get_container),
):
    """Analisa uma imagem enviada via multipart/form-data."""
    contents = await file.read()
    frame = _decode_bytes_to_image(contents)
    result = container.analyze_frame_uc.execute(frame, camera_id=camera_id, check_compliance=check_compliance)
    return _to_response_dto(result)


@router.post("/analyze/base64", response_model=FrameAnalysisResponse, tags=["Inferência"])
def analyze_base64_payload(
    payload: Base64AnalyzeRequest,
    container: Container = Depends(get_container),
):
    """Analisa uma imagem codificada em Base64."""
    data_str = payload.image_base64
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(data_str)
    except Exception:
        raise HTTPException(status_code=400, detail="String Base64 inválida.")

    frame = _decode_bytes_to_image(raw_bytes)
    result = container.analyze_frame_uc.execute(frame, camera_id=payload.camera_id, check_compliance=payload.check_compliance)
    return _to_response_dto(result)


@router.post("/analyze/annotated", tags=["Inferência"])
async def analyze_and_return_image(
    file: UploadFile = File(...),
    container: Container = Depends(get_container),
):
    """Processa a imagem e retorna o JPEG com as anotações visuais desenhadas."""
    contents = await file.read()
    frame = _decode_bytes_to_image(contents)
    detections = container.detector.detect(frame, skip_frames=1)
    annotated = draw_detections(frame, detections)
    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem anotada.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
