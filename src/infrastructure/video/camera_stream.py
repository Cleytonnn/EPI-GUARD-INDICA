"""Gerenciamento de streams de vídeo em tempo real (Webcam e RTSP)."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """Captura de vídeo assíncrona em thread dedicada com buffer de tamanho 1."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        reconnect_delay: int = 5,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._connected = False

    def start(self) -> bool:
        """Inicia a captura de vídeo."""
        if not self._connect():
            logger.error(f"❌ Falha inicial ao conectar à fonte: {self.source}")
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"📹 Stream de vídeo iniciado: {self.source}")
        return True

    def _connect(self) -> bool:
        with self._lock:
            if self._cap is not None:
                self._cap.release()

            if isinstance(self.source, int):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)

            if not self._cap.isOpened():
                self._connected = False
                return False

            if isinstance(self.source, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            self._connected = True
            return True

    def _capture_loop(self) -> None:
        """Loop de leitura executado em thread desacoplada."""
        while self._running:
            if not self._connected:
                time.sleep(self.reconnect_delay)
                self._connect()
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("⚠️ Perda de sinal no stream de vídeo.")
                self._connected = False
                continue

            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Lê o quadro mais recente disponível."""
        try:
            frame = self._queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        """Encerra a captura e libera recursos."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
        self._connected = False
        logger.info("📹 Stream de vídeo finalizado.")

    @property
    def is_connected(self) -> bool:
        return self._connected
