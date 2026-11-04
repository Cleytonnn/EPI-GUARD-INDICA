"""Adaptador de infraestrutura para captura de snapshots via APIs HTTP de CFTV/VMS.

Suporta Hikvision ISAPI, Intelbras, ONVIF HTTP Snapshot e endpoints REST JSON (Base64),
integrando-se aos serviços de domínio e armazenamento da Clean Architecture.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth
import yaml

from src.config.settings import ROOT_DIR
from src.core.interfaces.alert_storage import IAlertStorage
from src.core.interfaces.detector import IDetector
from src.core.interfaces.notifier import INotifier
from src.core.services.compliance_service import ComplianceService
from src.presentation.cli.visualizer import draw_detections

logger = logging.getLogger(__name__)


@dataclass
class RemoteCameraConfig:
    """Configuração de conexão de uma câmera remota via HTTP/API."""

    camera_id: str
    name: str
    url: str
    method: str = "GET"
    auth_type: str = "none"  # none, basic, digest, bearer, api_key
    username: str = ""
    password: str = ""
    token: str = ""
    api_key_header: str = ""
    headers: Optional[Dict[str, str]] = None
    body: Optional[Dict] = None
    interval: float = 2.0
    timeout: int = 10
    enabled: bool = True
    image_field: str = ""


class CameraAPIClient:
    """Cliente que consulta periodicamente snapshots de câmeras via HTTP e audita EPIs."""

    def __init__(
        self,
        cameras: List[RemoteCameraConfig],
        detector: IDetector,
        compliance_factory: Callable[[], ComplianceService],
        alert_storage: IAlertStorage,
        notifier: Optional[INotifier] = None,
        max_workers: int = 10,
    ):
        self.cameras = {c.camera_id: c for c in cameras if c.enabled}
        self.detector = detector
        self.compliance_factory = compliance_factory
        self.compliance_checkers: Dict[str, ComplianceService] = {}
        self.alert_storage = alert_storage
        self.notifier = notifier
        self.max_workers = max_workers

        self._running = False
        self._sessions: Dict[str, requests.Session] = {}
        self._stats = {
            "total_requests": 0,
            "total_errors": 0,
            "total_detections": 0,
            "total_violations": 0,
        }

        for cam_id, cam in self.cameras.items():
            session = requests.Session()
            session.headers.update({"User-Agent": "EPI-Guard-System/2.0"})

            if cam.auth_type == "basic":
                session.auth = HTTPBasicAuth(cam.username, cam.password)
            elif cam.auth_type == "digest":
                session.auth = HTTPDigestAuth(cam.username, cam.password)
            elif cam.auth_type == "bearer":
                session.headers["Authorization"] = f"Bearer {cam.token}"
            elif cam.auth_type == "api_key":
                header_name = cam.api_key_header or "X-API-Key"
                session.headers[header_name] = cam.token

            if cam.headers:
                session.headers.update(cam.headers)

            self._sessions[cam_id] = session

        logger.info(f"📹 {len(self.cameras)} câmera(s) remota(s) configurada(s) via API")

    def _get_compliance_service(self, camera_id: str) -> ComplianceService:
        if camera_id not in self.compliance_checkers:
            self.compliance_checkers[camera_id] = self.compliance_factory()
        return self.compliance_checkers[camera_id]

    def fetch_snapshot(self, camera_id: str) -> Optional[np.ndarray]:
        """Busca um snapshot de uma câmera via HTTP e decodifica para BGR."""
        cam = self.cameras.get(camera_id)
        if not cam:
            return None

        session = self._sessions[camera_id]

        try:
            if cam.method.upper() == "POST":
                response = session.post(cam.url, json=cam.body, timeout=cam.timeout)
            else:
                response = session.get(cam.url, timeout=cam.timeout)

            response.raise_for_status()
            self._stats["total_requests"] += 1

            content_type = response.headers.get("Content-Type", "")

            if "json" in content_type:
                data = response.json()
                image_data = None
                if cam.image_field:
                    image_data = data
                    for key in cam.image_field.split("."):
                        image_data = image_data[key]
                else:
                    for field in ("image", "snapshot", "frame", "data", "img", "picture"):
                        if field in data:
                            image_data = data[field]
                            break

                if image_data is None:
                    logger.error(f"❌ [{camera_id}] Campo de imagem não encontrado no JSON")
                    return None

                if isinstance(image_data, str) and "," in image_data:
                    image_data = image_data.split(",", 1)[1]

                img_bytes = base64.b64decode(image_data)
                img_array = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error(f"❌ [{camera_id}] Não foi possível decodificar a imagem recebida")
                return None

            return frame

        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ [{camera_id}] Timeout ao buscar snapshot")
            self._stats["total_errors"] += 1
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔌 [{camera_id}] Erro de conexão: {cam.url}")
            self._stats["total_errors"] += 1
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            logger.warning(f"❌ [{camera_id}] HTTP {code}: {e}")
            self._stats["total_errors"] += 1
        except Exception as e:
            logger.error(f"❌ [{camera_id}] Erro inesperado: {e}")
            self._stats["total_errors"] += 1

        return None

    def _process_camera(self, camera_id: str) -> None:
        frame = self.fetch_snapshot(camera_id)
        if frame is None:
            return

        detections = self.detector.detect(frame, skip_frames=1)
        if detections:
            self._stats["total_detections"] += len(detections)

        checker = self._get_compliance_service(camera_id)
        violations = checker.check(detections)

        if violations:
            self._stats["total_violations"] += len(violations)
            annotated = draw_detections(frame, detections)
            for viol in violations:
                logger.warning(f"🚨 [{camera_id}] {viol.description}")
                saved_path = self.alert_storage.save_image(annotated, viol)
                if self.notifier:
                    self.notifier.notify(viol, image_path=saved_path)

    def _camera_loop(self, camera_id: str) -> None:
        cam = self.cameras[camera_id]
        logger.info(f"📹 [{camera_id}] Iniciando monitoramento: {cam.name}")

        while self._running:
            self._process_camera(camera_id)
            time.sleep(cam.interval)

        logger.info(f"📹 [{camera_id}] Monitoramento encerrado.")

    def start(self) -> None:
        """Inicia o pool de threads para monitorar todas as câmeras habilitadas."""
        self._running = True

        logger.info("=" * 60)
        logger.info("🚀 MONITORAMENTO VIA API DE CÂMERAS INICIADO")
        logger.info(f"📹 Câmeras ativas: {len(self.cameras)}")
        for cam_id, cam in self.cameras.items():
            logger.info(f"   [{cam_id}] {cam.name} - a cada {cam.interval}s")
        logger.info("=" * 60)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for cam_id in self.cameras:
                executor.submit(self._camera_loop, cam_id)

            try:
                while self._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Interrupção detectada. Encerrando...")
                self.stop()

        self._print_stats()

    def stop(self) -> None:
        """Encerra o monitoramento e fecha sessões HTTP."""
        self._running = False
        for session in self._sessions.values():
            session.close()

    def _print_stats(self) -> None:
        logger.info("=" * 60)
        logger.info("📊 ESTATÍSTICAS DA SESSÃO")
        logger.info(f"   Requisições: {self._stats['total_requests']}")
        logger.info(f"   Erros: {self._stats['total_errors']}")
        logger.info(f"   Detecções: {self._stats['total_detections']}")
        logger.info(f"   Violações: {self._stats['total_violations']}")
        logger.info(f"   Imagens salvas: {getattr(self.alert_storage, 'total_saved', 0)}")
        logger.info("=" * 60)


def load_cameras_config(config_path: Optional[str | Path] = None) -> List[RemoteCameraConfig]:
    """Carrega a lista de câmeras remotas a partir de config/cameras.yaml."""
    target = Path(config_path) if config_path is not None else ROOT_DIR / "config" / "cameras.yaml"
    if not target.exists():
        logger.error(f"❌ Arquivo de câmeras não encontrado: {target}")
        return []

    with open(target, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cameras: List[RemoteCameraConfig] = []
    for cam_data in data.get("cameras", []):
        cameras.append(
            RemoteCameraConfig(
                camera_id=cam_data["camera_id"],
                name=cam_data.get("name", cam_data["camera_id"]),
                url=cam_data["url"],
                method=cam_data.get("method", "GET"),
                auth_type=cam_data.get("auth_type", "none"),
                username=cam_data.get("username", ""),
                password=cam_data.get("password", ""),
                token=cam_data.get("token", ""),
                api_key_header=cam_data.get("api_key_header", ""),
                headers=cam_data.get("headers"),
                body=cam_data.get("body"),
                interval=float(cam_data.get("interval", 2.0)),
                timeout=int(cam_data.get("timeout", 10)),
                enabled=bool(cam_data.get("enabled", True)),
                image_field=cam_data.get("image_field", ""),
            )
        )

    return cameras

