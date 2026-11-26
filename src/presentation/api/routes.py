"""Rotas e controladores REST da aplicação."""

from __future__ import annotations

import base64
import binascii
import hmac
import re

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, Security, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.security import APIKeyHeader

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
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_DATA_URI_RE = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,", re.IGNORECASE)


def get_container() -> Container:
    """Dependency provider do container de injeção."""
    from src.presentation.api.app import app_container

    return app_container


def require_api_key(
    request: Request,
    header_key: str | None = Security(_api_key_header),
    container: Container = Depends(get_container),
) -> None:
    """Exige API Key opcional (X-API-Key ou Authorization: Bearer) quando configurada."""
    expected = container.settings.api.api_key
    if not expected:
        return

    provided = header_key
    if not provided:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()

    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="API Key inválida ou ausente.")


def _max_upload_bytes(container: Container) -> int:
    return max(1, int(container.settings.api.max_upload_mb)) * 1024 * 1024


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo excede o limite máximo permitido ({max_bytes // (1024 * 1024)} MB).",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")
    return contents


def _decode_base64_payload(data_str: str, max_bytes: int) -> bytes:
    cleaned = data_str.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Payload Base64 vazio.")

    if cleaned.startswith("data:") or "," in cleaned:
        match = _DATA_URI_RE.match(cleaned)
        if not match:
            raise HTTPException(status_code=400, detail="Prefixo data URI de imagem inválido.")
        cleaned = cleaned[match.end() :]

    cleaned = re.sub(r"\s+", "", cleaned)
    max_b64_len = (max_bytes * 4 // 3) + 128
    if len(cleaned) > max_b64_len:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem Base64 excede o limite máximo permitido ({max_bytes // (1024 * 1024)} MB).",
        )

    try:
        raw_bytes = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="String Base64 inválida.") from exc

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Payload Base64 vazio.")
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem excede o limite máximo permitido ({max_bytes // (1024 * 1024)} MB).",
        )
    return raw_bytes


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
        model=str(getattr(det, "model_path", "unknown")),
        device=det.device,
        classes_count=len(det.class_names),
    )


@router.get("/stats", tags=["Monitoramento"], dependencies=[Depends(require_api_key)])
def get_system_stats(container: Container = Depends(get_container)):
    """Retorna estatísticas operacionais de alertas e processamento."""
    return {
        "imagens_salvas": getattr(container.alert_storage, "total_saved", 0),
        "emails_enviados": getattr(container.email_notifier, "total_sent", 0),
    }


def _analyze_bytes_sync(
    container: Container,
    img_bytes: bytes,
    camera_id: str | None,
    check_compliance: bool,
) -> FrameAnalysisResponse:
    frame = _decode_bytes_to_image(img_bytes)
    result = container.analyze_frame_uc.execute(frame, camera_id=camera_id, check_compliance=check_compliance)
    return _to_response_dto(result)


def _analyze_annotated_sync(container: Container, img_bytes: bytes) -> bytes:
    frame = _decode_bytes_to_image(img_bytes)
    detections = container.detector.detect(frame, skip_frames=1)
    annotated = draw_detections(frame, detections)
    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Falha ao codificar imagem anotada.")
    return encoded.tobytes()


@router.post(
    "/analyze/upload",
    response_model=FrameAnalysisResponse,
    tags=["Inferência"],
    dependencies=[Depends(require_api_key)],
)
async def analyze_file_upload(
    file: UploadFile = File(..., description="Arquivo de imagem (JPEG/PNG)"),
    camera_id: str = Form("cam_upload", description="Identificador da câmera"),
    check_compliance: bool = Form(True, description="Verificar conformidade temporal"),
    container: Container = Depends(get_container),
):
    """Analisa uma imagem enviada via multipart/form-data."""
    contents = await _read_upload_limited(file, _max_upload_bytes(container))
    return await run_in_threadpool(_analyze_bytes_sync, container, contents, camera_id, check_compliance)


@router.post(
    "/analyze/base64",
    response_model=FrameAnalysisResponse,
    tags=["Inferência"],
    dependencies=[Depends(require_api_key)],
)
async def analyze_base64_payload(
    payload: Base64AnalyzeRequest,
    container: Container = Depends(get_container),
):
    """Analisa uma imagem codificada em Base64."""
    raw_bytes = _decode_base64_payload(payload.image_base64, _max_upload_bytes(container))
    return await run_in_threadpool(
        _analyze_bytes_sync, container, raw_bytes, payload.camera_id, payload.check_compliance
    )


@router.post("/analyze/annotated", tags=["Inferência"], dependencies=[Depends(require_api_key)])
async def analyze_and_return_image(
    file: UploadFile = File(...),
    container: Container = Depends(get_container),
):
    """Processa a imagem e retorna o JPEG com as anotações visuais desenhadas."""
    contents = await _read_upload_limited(file, _max_upload_bytes(container))
    jpeg_bytes = await run_in_threadpool(_analyze_annotated_sync, container, contents)
    return Response(content=jpeg_bytes, media_type="image/jpeg")
