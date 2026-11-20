"""Widget de Reprodução de Vídeo incorporado usando o motor libVLC.

Permite reproduzir arquivos de vídeo locais (.265, .mp4, etc.) ou fluxos RTSP
com aceleração por hardware nativa do VLC diretamente na janela do CustomTkinter.
"""

from __future__ import annotations

import logging
import platform
import tkinter as tk
from pathlib import Path

import customtkinter as ctk

logger = logging.getLogger(__name__)

try:
    import vlc

    HAS_VLC = True
except Exception:
    HAS_VLC = False


class EmbeddedVLCPlayer(ctk.CTkFrame):
    """Componente gráfico que renderiza o reprodutor VLC nativo embutido no Tkinter."""

    def __init__(self, master, width: int = 560, height: int = 340, **kwargs):
        super().__init__(master, width=width, height=height, **kwargs)
        self.width = width
        self.height = height

        self._instance: vlc.Instance | None = None
        self._player: vlc.MediaPlayer | None = None
        self._current_media_path: str | None = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Frame nativo Tkinter para receber o handle de janela (HWND) do Windows
        self.video_frame = tk.Frame(self, bg="#0c1015", width=width, height=height)
        self.video_frame.grid(row=0, column=0, sticky="nsew")

        if HAS_VLC:
            try:
                # Inicializar instância do libVLC com aceleração por hardware
                self._instance = vlc.Instance("--no-xlib", "--quiet")
                self._player = self._instance.media_player_new()
            except Exception as e:
                logger.warning(f"Não foi possível inicializar instância do VLC: {e}")

    def attach_window(self) -> None:
        """Vincula o handle da janela ao player VLC (Windows: HWND)."""
        if not self._player:
            return

        self.update_idletasks()
        win_id = self.video_frame.winfo_id()
        if platform.system() == "Windows":
            self._player.set_hwnd(win_id)
        elif platform.system() == "Darwin":
            self._player.set_nsobject(win_id)
        else:
            self._player.set_xwindow(win_id)

    def load_media(self, media_path: str | Path) -> bool:
        """Carrega um arquivo ou URL para reprodução."""
        if not self._instance or not self._player:
            return False

        path_str = (
            str(Path(media_path).resolve())
            if not str(media_path).startswith(("rtsp://", "http://"))
            else str(media_path)
        )
        self._current_media_path = path_str

        try:
            media = self._instance.media_new(path_str)
            self._player.set_media(media)
            self.attach_window()
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar mídia no VLC: {e}")
            return False

    def play(self) -> None:
        """Inicia ou retoma a reprodução."""
        if self._player:
            self.attach_window()
            self._player.play()

    def pause(self) -> None:
        """Pausa ou alterna a reprodução."""
        if self._player:
            self._player.pause()

    def stop(self) -> None:
        """Interrompe a reprodução."""
        if self._player:
            self._player.stop()

    def seek(self, position_ratio: float) -> None:
        """Define a posição de reprodução (0.0 a 1.0)."""
        if self._player:
            self._player.set_position(max(0.0, min(1.0, position_ratio)))

    def is_playing(self) -> bool:
        """Indica se o player está reproduzindo no momento."""
        return bool(self._player and self._player.is_playing())

    def get_time(self) -> int:
        """Retorna o tempo atual em milissegundos."""
        return self._player.get_time() if self._player else 0

    def get_length(self) -> int:
        """Retorna a duração total em milissegundos."""
        return self._player.get_length() if self._player else 0
