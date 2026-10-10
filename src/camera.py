"""Módulo de captura de vídeo para o sistema de detecção de EPI.

Gerencia a captura de frames de webcam, câmeras IP (RTSP) ou arquivos de vídeo,
com reconexão automática e threading para performance.
"""

import cv2
import threading
import time
import logging
from typing import Optional, Union
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class CameraManager:
    """Gerenciador de captura de vídeo com threading e reconexão automática.
    
    Suporta webcam local, câmeras IP via RTSP e arquivos de vídeo.
    Utiliza uma thread dedicada para captura contínua de frames,
    evitando bloqueio no loop principal de detecção.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        reconnect_delay: int = 5,
        queue_size: int = 2,
    ):
        """Inicializa o gerenciador de câmera.
        
        Args:
            source: Fonte de vídeo (0=webcam, 'rtsp://...' para IP, 'video.mp4' para arquivo)
            width: Largura desejada do frame
            height: Altura desejada do frame
            fps: FPS desejado (apenas para webcam)
            reconnect_delay: Segundos entre tentativas de reconexão
            queue_size: Tamanho máximo da fila de frames (menor = menos latência)
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.reconnect_delay = reconnect_delay
        self.queue_size = queue_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: Queue = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_file = isinstance(source, str) and not source.startswith("rtsp")

    def start(self) -> bool:
        """Inicia a captura de vídeo em uma thread dedicada.
        
        Returns:
            True se a câmera foi aberta com sucesso, False caso contrário.
        """
        if not self._open_camera():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"📹 Captura iniciada: {self.source}")
        return True

    def _open_camera(self) -> bool:
        """Abre a conexão com a câmera/vídeo."""
        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                logger.error(f"❌ Não foi possível abrir a fonte de vídeo: {self.source}")
                return False

            # Configurar resolução e FPS (apenas para webcam/câmera)
            if not self._is_file:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"📹 Câmera aberta: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
            return True

        except Exception as e:
            logger.error(f"❌ Erro ao abrir câmera: {e}")
            return False

    def _capture_loop(self):
        """Loop de captura executado em thread dedicada."""
        while self._running:
            with self._lock:
                if self._cap is None or not self._cap.isOpened():
                    if not self._reconnect():
                        time.sleep(self.reconnect_delay)
                        continue

                ret, frame = self._cap.read()

            if not ret:
                if self._is_file:
                    logger.info("📹 Fim do arquivo de vídeo.")
                    self._running = False
                    break
                else:
                    logger.warning("⚠️ Falha na leitura do frame. Tentando reconectar...")
                    self._reconnect()
                    continue

            # Se a fila está cheia, descarta o frame mais antigo (manter baixa latência)
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
            self._frame_queue.put(frame)

    def _reconnect(self) -> bool:
        """Tenta reconectar com a câmera."""
        logger.info(f"🔄 Tentando reconectar em {self.reconnect_delay}s...")
        self.release_camera()
        time.sleep(self.reconnect_delay)
        return self._open_camera()

    def read(self) -> Optional[any]:
        """Lê o frame mais recente da fila.
        
        Returns:
            Frame numpy array ou None se não houver frame disponível.
        """
        try:
            return self._frame_queue.get(timeout=1.0)
        except Empty:
            return None

    def is_running(self) -> bool:
        """Verifica se a captura está ativa."""
        return self._running

    def release_camera(self):
        """Libera o recurso da câmera."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def stop(self):
        """Para a captura e libera todos os recursos."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.release_camera()
        logger.info("📹 Captura de vídeo encerrada.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
