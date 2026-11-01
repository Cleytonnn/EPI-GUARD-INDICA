"""Leitor de vídeo universal com suporte nativo a .265, .h265, .hevc, .mp4, .avi, etc."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from src.core.interfaces.video_reader import IVideoReader

logger = logging.getLogger(__name__)


class UniversalVideoReader(IVideoReader):
    """Implementa IVideoReader com decodificação direta de .265 e formatos padrão.
    
    Verifica a integridade inicial do arquivo para evitar travamentos em arquivos corrompidos
    preenchidos apenas por bytes nulos (0x00).
    """

    def __init__(self, video_path: str | Path):
        self.video_path = str(video_path)
        self.backend: Optional[str] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._container = None
        self._stream = None
        self._frame_generator = None

        self._fps: float = 25.0
        self._total_frames: int = 0
        self._width: int = 0
        self._height: int = 0
        self._codec_name: str = "desconhecido"

        self._init_reader()

    def _init_reader(self) -> None:
        # 0. Verificação de integridade: checar arquivos de DVR corrompidos (100% zeros)
        try:
            with open(self.video_path, "rb") as fp:
                head = fp.read(65536)
                if not head or not any(b != 0 for b in head):
                    logger.error(
                        f"⚠️ ARQUIVO CORROMPIDO OU INCOMPLETO: {self.video_path}\n"
                        f"   O arquivo contém apenas bytes nulos (0x00). O download/gravação foi interrompido antes do término."
                    )
                    self.backend = None
                    return
        except Exception as e:
            logger.error(f"❌ Erro ao abrir arquivo para leitura: {e}")
            self.backend = None
            return

        suffix = Path(self.video_path).suffix.lower()

        # 1. Tentar OpenCV com FFMPEG nativo primeiro (sempre gera frames limpos sem macroblocos verdes em .265/.mp4)
        if self._try_opencv():
            return

        # 2. Se for arquivo de CFTV (.265/.hevc) e o VLC estiver disponível, usar motor VLC para remux limpo
        if suffix in {".265", ".h265", ".hevc"}:
            if self._try_vlc_remux():
                return

        # 3. Fallback: PyAV
        if suffix in {".265", ".h265", ".hevc"}:
            if self._try_pyav(force_format="hevc"):
                return
        if self._try_pyav():
            return

        logger.error(f"❌ Não foi possível abrir o arquivo com nenhum decodificador: {self.video_path}")
        self.backend = None

    def _try_vlc_remux(self) -> bool:
        """Tenta usar o VLC para reempacotar stream cru de CFTV para contêiner MP4 e abrir."""
        try:
            from src.infrastructure.video.vlc_service import VLCService
            if not VLCService.is_available():
                return False

            import tempfile
            temp_mp4 = Path(tempfile.gettempdir()) / f"vlc_clean_{Path(self.video_path).stem}.mp4"
            remuxed = VLCService.remux_to_clean_mp4(self.video_path, temp_mp4)
            if remuxed and remuxed.exists():
                cap = cv2.VideoCapture(str(remuxed), cv2.CAP_FFMPEG)
                if cap.isOpened():
                    self._cap = cap
                    self.backend = "vlc-remux"
                    self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._codec_name = "vlc-mp4"
                    logger.info(f"🎬 Vídeo .{Path(self.video_path).suffix} recuperado com sucesso via motor VLC!")
                    return True
        except Exception as e:
            logger.debug(f"VLC remux falhou: {e}")
        return False

    def _try_opencv(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._cap = cap
                    self.backend = "opencv"
                    self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self._codec_name = "opencv-ffmpeg"
                    logger.debug(f"📹 Aberto via OpenCV FFMPEG: {self.video_path}")
                    return True
                cap.release()
        except Exception as e:
            logger.debug(f"OpenCV falhou para {self.video_path}: {e}")
        return False

    def _try_pyav(self, force_format: Optional[str] = None) -> bool:
        try:
            import av
            if force_format:
                container = av.open(self.video_path, format=force_format)
            else:
                container = av.open(self.video_path)

            if container.streams.video:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                self._container = container
                self._stream = stream
                self.backend = "pyav"
                self._codec_name = stream.codec_context.name
                self._fps = float(stream.average_rate) if stream.average_rate else 25.0
                self._total_frames = int(stream.frames) if stream.frames else 0
                self._width = stream.width or 0
                self._height = stream.height or 0
                self._frame_generator = container.decode(video=0)
                logger.info(
                    f"🎞️ Vídeo .{Path(self.video_path).suffix} aberto via PyAV: "
                    f"{Path(self.video_path).name} (codec: {self._codec_name.upper()}, {self._width}x{self._height})"
                )
                return True
        except Exception as e:
            logger.debug(f"PyAV falhou para {self.video_path} (format={force_format}): {e}")
        return False

    def is_opened(self) -> bool:
        return self.backend is not None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.backend == "opencv" and self._cap is not None:
            return self._cap.read()
        elif self.backend == "pyav" and self._frame_generator is not None:
            try:
                frame = next(self._frame_generator)
                img = frame.to_ndarray(format="bgr24")
                return True, img
            except StopIteration:
                return False, None
            except Exception as e:
                logger.debug(f"Fim de leitura PyAV: {e}")
                return False, None
        return False, None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec_name(self) -> str:
        return self._codec_name
