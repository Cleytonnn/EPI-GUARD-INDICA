"""Rotas e controladores REST da aplicação."""

from __future__ import annotations

import base64
import binascii
import hmac

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

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


def get_container() -> Container:
    """Dependency provider do container de injeção."""
    from src.presentation.api.app import app_container

    if app_container is None:
        raise HTTPException(status_code=503, detail="Serviço ainda não inicializado.")
    return app_container


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    container: Container = Depends(get_container),
) -> None:
    """Exige o header X-API-Key quando API_KEY estiver definido no .env."""
    expected = getattr(container.settings.api, "api_key", None)
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="API key ausente ou inválida.")


router = APIRouter()
protected = [Depends(require_api_key)]


def _max_bytes(container: Container) -> int:
    return int(float(getattr(container.settings.api, "max_upload_mb", 10.0)) * 1024 * 1024)


async def _read_upload(file: UploadFile, container: Container) -> bytes:
    """Lê o upload com limite de tamanho (antes não havia limite → risco de estouro de memória)."""
    limit = _max_bytes(container)
    contents = await file.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(status_code=413, detail=f"Imagem maior que o limite de {limit // (1024 * 1024)} MB.")
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    return contents


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
        frame_violations=len(result.frame_violations),
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
        model=str(getattr(det, "model_path", "desconhecido")),
        device=str(getattr(det, "device", "desconhecido")),
        classes_count=len(getattr(det, "class_names", {}) or {}),
    )


@router.get("/stats", tags=["Monitoramento"], dependencies=protected)
def get_system_stats(container: Container = Depends(get_container)):
    """Retorna estatísticas operacionais de alertas e processamento."""
    return {
        "imagens_salvas": getattr(container.alert_storage, "total_saved", 0),
        "emails_enviados": getattr(container.email_notifier, "total_sent", 0),
    }


@router.post("/analyze/upload", response_model=FrameAnalysisResponse, tags=["Inferência"], dependencies=protected)
async def analyze_file_upload(
    file: UploadFile = File(..., description="Arquivo de imagem (JPEG/PNG)"),
    camera_id: str = Form("cam_upload", description="Identificador da câmera"),
    check_compliance: bool = Form(True, description="Verificar conformidade temporal"),
    container: Container = Depends(get_container),
):
    """Analisa uma imagem enviada via multipart/form-data."""
    contents = await _read_upload(file, container)
    frame = _decode_bytes_to_image(contents)
    # Inferência é pesada e síncrona: roda fora do event loop para não travar as demais requisições
    result = await run_in_threadpool(
        container.analyze_frame_uc.execute, frame, camera_id=camera_id, check_compliance=check_compliance
    )
    return _to_response_dto(result)


@router.post("/analyze/base64", response_model=FrameAnalysisResponse, tags=["Inferência"], dependencies=protected)
def analyze_base64_payload(
    payload: Base64AnalyzeRequest,
    container: Container = Depends(get_container),
):
    """Analisa uma imagem codificada em Base64."""
    data_str = payload.image_base64
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    data_str = "".join(data_str.split())  # remove quebras de linha/espaços

    limit = _max_bytes(container)
    if len(data_str) * 3 // 4 > limit:
        raise HTTPException(status_code=413, detail=f"Imagem maior que o limite de {limit // (1024 * 1024)} MB.")

    try:
        raw_bytes = base64.b64decode(data_str, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="String Base64 inválida.") from exc

    frame = _decode_bytes_to_image(raw_bytes)
    result = container.analyze_frame_uc.execute(
        frame, camera_id=payload.camera_id, check_compliance=payload.check_compliance
    )
    return _to_response_dto(result)


@router.post("/analyze/annotated", tags=["Inferência"], dependencies=protected)
async def analyze_and_return_image(
    file: UploadFile = File(...),
    container: Container = Depends(get_container),
):
    """Processa a imagem e retorna o JPEG com as anotações visuais desenhadas."""
    contents = await _read_upload(file, container)
    frame = _decode_bytes_to_image(contents)
    detections = await run_in_threadpool(container.detector.detect, frame, 1)
    annotated = draw_detections(frame, detections)
    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem anotada.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")
