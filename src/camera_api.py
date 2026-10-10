"""Cliente para APIs de sistemas de câmeras.

Conecta em sistemas de CFTV/VMS via API para capturar
snapshots periódicos e enviar para detecção de EPI.

Suporta:
- API genérica (HTTP GET retorna imagem)
- Hikvision ISAPI
- Intelbras
- ONVIF snapshot
- Qualquer sistema que forneça imagem via HTTP

Uso:
    python -m src.camera_api
    python -m src.camera_api --config config/cameras.yaml
"""

import os
import sys
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import cv2
import yaml
import numpy as np
import requests
from requests.auth import HTTPBasicAuth, HTTPDigestAuth

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.detector import PPEDetector
from src.compliance import ComplianceChecker
from src.alerter import AlertManager
from src.utils import draw_detections, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuração de uma câmera remota."""
    camera_id: str              # ID único da câmera
    name: str                   # Nome descritivo
    url: str                    # URL da API/snapshot
    method: str = "GET"         # Método HTTP (GET ou POST)
    auth_type: str = "none"     # Tipo de autenticação: none, basic, digest, bearer, api_key
    username: str = ""          # Usuário (para basic/digest)
    password: str = ""          # Senha (para basic/digest)
    token: str = ""             # Token (para bearer/api_key)
    api_key_header: str = ""    # Nome do header para API key
    headers: Dict = None        # Headers adicionais
    body: Dict = None           # Body para requisições POST
    interval: float = 2.0       # Intervalo entre capturas (segundos)
    timeout: int = 10           # Timeout da requisição (segundos)
    enabled: bool = True        # Se a câmera está ativa
    image_field: str = ""       # Campo JSON que contém a imagem (se resposta for JSON)


class CameraAPIClient:
    """Cliente que conecta na API do sistema de câmeras.
    
    Faz requisições periódicas para capturar snapshots e
    envia para o pipeline de detecção de EPI.
    """

    def __init__(
        self,
        cameras: List[CameraConfig],
        detector: PPEDetector,
        compliance_checkers: Dict[str, ComplianceChecker],
        alerter: AlertManager,
        max_workers: int = 10,
    ):
        """Inicializa o cliente de API de câmeras.
        
        Args:
            cameras: Lista de configurações de câmeras.
            detector: Detector de EPI.
            compliance_checkers: Dict de checkers por câmera.
            alerter: Gerenciador de alertas.
            max_workers: Máximo de threads para requisições paralelas.
        """
        self.cameras = {c.camera_id: c for c in cameras if c.enabled}
        self.detector = detector
        self.compliance_checkers = compliance_checkers
        self.alerter = alerter
        self.max_workers = max_workers

        self._running = False
        self._sessions: Dict[str, requests.Session] = {}
        self._stats = {
            "total_requests": 0,
            "total_errors": 0,
            "total_detections": 0,
            "total_violations": 0,
        }

        # Criar sessões HTTP reutilizáveis (connection pooling)
        for cam_id, cam in self.cameras.items():
            session = requests.Session()
            session.headers.update({"User-Agent": "PPE-Detection-System/1.0"})

            # Configurar autenticação
            if cam.auth_type == "basic":
                session.auth = HTTPBasicAuth(cam.username, cam.password)
            elif cam.auth_type == "digest":
                session.auth = HTTPDigestAuth(cam.username, cam.password)
            elif cam.auth_type == "bearer":
                session.headers["Authorization"] = f"Bearer {cam.token}"
            elif cam.auth_type == "api_key":
                header_name = cam.api_key_header or "X-API-Key"
                session.headers[header_name] = cam.token

            # Headers adicionais
            if cam.headers:
                session.headers.update(cam.headers)

            self._sessions[cam_id] = session

        logger.info(f"📹 {len(self.cameras)} câmera(s) configurada(s) via API")

    def _get_compliance_checker(self, camera_id: str) -> ComplianceChecker:
        """Obtém ou cria um ComplianceChecker para uma câmera."""
        if camera_id not in self.compliance_checkers:
            self.compliance_checkers[camera_id] = ComplianceChecker()
        return self.compliance_checkers[camera_id]

    def fetch_snapshot(self, camera_id: str) -> Optional[np.ndarray]:
        """Busca um snapshot de uma câmera via API.
        
        Args:
            camera_id: ID da câmera.
            
        Returns:
            Frame numpy array (BGR) ou None em caso de erro.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return None

        session = self._sessions[camera_id]

        try:
            # Fazer requisição
            if cam.method.upper() == "POST":
                response = session.post(
                    cam.url,
                    json=cam.body,
                    timeout=cam.timeout,
                )
            else:
                response = session.get(cam.url, timeout=cam.timeout)

            response.raise_for_status()
            self._stats["total_requests"] += 1

            # Interpretar resposta
            content_type = response.headers.get("Content-Type", "")

            if "image" in content_type:
                # Resposta é imagem direta (JPEG, PNG)
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            elif "json" in content_type:
                # Resposta é JSON com imagem em Base64
                import base64
                data = response.json()

                # Tentar encontrar o campo da imagem
                image_data = None
                if cam.image_field:
                    # Campo especificado na config
                    image_data = data
                    for key in cam.image_field.split("."):
                        image_data = image_data[key]
                else:
                    # Procurar campos comuns
                    for field in ["image", "snapshot", "frame", "data", "img", "picture"]:
                        if field in data:
                            image_data = data[field]
                            break

                if image_data is None:
                    logger.error(f"❌ [{camera_id}] Campo de imagem não encontrado no JSON")
                    return None

                # Remover prefixo data:image/... se presente
                if isinstance(image_data, str) and "," in image_data:
                    image_data = image_data.split(",", 1)[1]

                img_bytes = base64.b64decode(image_data)
                img_array = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            else:
                # Tentar interpretar como imagem mesmo assim
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                logger.error(f"❌ [{camera_id}] Não foi possível decodificar a imagem")
                return None

            return frame

        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ [{camera_id}] Timeout ao buscar snapshot")
            self._stats["total_errors"] += 1
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔌 [{camera_id}] Erro de conexão: {cam.url}")
            self._stats["total_errors"] += 1
        except requests.exceptions.HTTPError as e:
            logger.warning(f"❌ [{camera_id}] HTTP {e.response.status_code}: {e}")
            self._stats["total_errors"] += 1
        except Exception as e:
            logger.error(f"❌ [{camera_id}] Erro inesperado: {e}")
            self._stats["total_errors"] += 1

        return None

    def _process_camera(self, camera_id: str):
        """Processa uma câmera: busca snapshot → detecta → verifica."""
        frame = self.fetch_snapshot(camera_id)
        if frame is None:
            return

        # Detectar EPIs
        detections = self.detector.detect(frame, skip_frames=1)
        if detections:
            self._stats["total_detections"] += len(detections)

        # Verificar compliance
        checker = self._get_compliance_checker(camera_id)
        violations = checker.check(detections)

        # Processar alertas
        if violations:
            self._stats["total_violations"] += len(violations)
            annotated = draw_detections(frame, detections)
            for v in violations:
                logger.warning(f"🚨 [{camera_id}] {v.description}")
                self.alerter.process_violation(annotated, v)

    def _camera_loop(self, camera_id: str):
        """Loop de captura para uma câmera individual."""
        cam = self.cameras[camera_id]
        logger.info(f"📹 [{camera_id}] Iniciando monitoramento: {cam.name}")

        while self._running:
            self._process_camera(camera_id)
            time.sleep(cam.interval)

        logger.info(f"📹 [{camera_id}] Monitoramento encerrado.")

    def start(self):
        """Inicia o monitoramento de todas as câmeras."""
        self._running = True

        logger.info("=" * 60)
        logger.info("🚀 MONITORAMENTO VIA API INICIADO")
        logger.info(f"📹 Câmeras: {len(self.cameras)}")
        for cam_id, cam in self.cameras.items():
            logger.info(f"   [{cam_id}] {cam.name} - a cada {cam.interval}s")
        logger.info("=" * 60)

        # Uma thread por câmera
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._camera_loop, cam_id): cam_id
                for cam_id in self.cameras
            }

            try:
                # Bloqueia até Ctrl+C
                while self._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Ctrl+C detectado. Encerrando...")
                self._running = False

        self._print_stats()

    def stop(self):
        """Para o monitoramento."""
        self._running = False
        # Fechar sessões HTTP
        for session in self._sessions.values():
            session.close()

    def _print_stats(self):
        """Imprime estatísticas finais."""
        logger.info("=" * 60)
        logger.info("📊 ESTATÍSTICAS DA SESSÃO")
        logger.info(f"   Requisições: {self._stats['total_requests']}")
        logger.info(f"   Erros: {self._stats['total_errors']}")
        logger.info(f"   Detecções: {self._stats['total_detections']}")
        logger.info(f"   Violações: {self._stats['total_violations']}")
        alert_stats = self.alerter.get_stats()
        logger.info(f"   Imagens salvas: {alert_stats['imagens_salvas']}")
        logger.info(f"   Emails enviados: {alert_stats['emails_enviados']}")
        logger.info("=" * 60)


def load_cameras_config(config_path: str = None) -> List[CameraConfig]:
    """Carrega configuração de câmeras do YAML."""
    if config_path is None:
        config_path = ROOT_DIR / "config" / "cameras.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        logger.error(f"❌ Arquivo de câmeras não encontrado: {config_path}")
        logger.info("📋 Crie o arquivo config/cameras.yaml com as câmeras da empresa.")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cameras = []
    for cam_data in data.get("cameras", []):
        cameras.append(CameraConfig(
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
            interval=cam_data.get("interval", 2.0),
            timeout=cam_data.get("timeout", 10),
            enabled=cam_data.get("enabled", True),
            image_field=cam_data.get("image_field", ""),
        ))

    return cameras


def main():
    """Função principal."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Monitor de EPI via API de Câmeras")
    parser.add_argument("--config", "-c", default=None, help="Arquivo de configuração de câmeras")
    parser.add_argument("--settings", "-s", default=None, help="Arquivo settings.yaml")
    args = parser.parse_args()

    # Carregar configurações
    settings_path = args.settings or ROOT_DIR / "config" / "settings.yaml"
    settings = {}
    if Path(settings_path).exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}

    det_cfg = settings.get("detection", {})
    alert_cfg = settings.get("alerts", {})

    # Carregar câmeras
    cameras = load_cameras_config(args.config)
    if not cameras:
        logger.error("❌ Nenhuma câmera configurada. Saindo.")
        return

    # Inicializar componentes
    detector = PPEDetector(
        model_path=det_cfg.get("model_path", "models/best.pt"),
        confidence=det_cfg.get("confidence", 0.4),
        iou_threshold=det_cfg.get("iou_threshold", 0.45),
        imgsz=det_cfg.get("imgsz", 640),
        track=False,
    )

    alerter = AlertManager(
        save_images=alert_cfg.get("save_images", True),
        save_dir=alert_cfg.get("save_dir", "alertas"),
        send_email=alert_cfg.get("send_email", False),
        cooldown_seconds=alert_cfg.get("cooldown_seconds", 60),
        image_quality=alert_cfg.get("image_quality", 85),
    )

    # Iniciar cliente
    client = CameraAPIClient(
        cameras=cameras,
        detector=detector,
        compliance_checkers={},
        alerter=alerter,
    )

    client.start()


if __name__ == "__main__":
    main()
