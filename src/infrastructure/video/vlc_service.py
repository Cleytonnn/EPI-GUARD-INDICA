"""Serviço de integração com o VLC Media Player.

Fornece localização de binários, execução externa do VLC e verificação
de disponibilidade do runtime libVLC no Windows.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

VLC_SEARCH_PATHS = [
    Path(r"C:\Program Files\VideoLAN\VLC\vlc.exe"),
    Path(r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"),
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "VideoLAN" / "VLC" / "vlc.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "VideoLAN" / "VLC" / "vlc.exe",
]


class VLCService:
    """Gerencia a integração com o reprodutor nativo e biblioteca do VLC."""

    @staticmethod
    def get_vlc_executable() -> Path | None:
        """Localiza o executável vlc.exe no sistema."""
        which_path = shutil.which("vlc")
        if which_path:
            p = Path(which_path)
            if p.exists():
                return p

        for candidate in VLC_SEARCH_PATHS:
            if candidate.exists():
                return candidate

        return None

    @classmethod
    def is_available(cls) -> bool:
        """Verifica se o VLC executável está disponível."""
        return cls.get_vlc_executable() is not None

    @classmethod
    def is_libvlc_available(cls) -> bool:
        """Verifica se o binding python-vlc consegue carregar a libvlc."""
        try:
            import vlc

            version = vlc.libvlc_get_version()
            return version is not None
        except Exception:
            return False

    @classmethod
    def open_video(cls, video_path: str | Path, start_time_seconds: float | None = None) -> bool:
        """Abre o arquivo de vídeo ou fluxo RTSP no VLC Media Player nativo."""
        vlc_exe = cls.get_vlc_executable()
        if not vlc_exe:
            logger.warning("VLC não encontrado no sistema.")
            return False

        target = (
            str(Path(video_path).resolve())
            if not str(video_path).startswith(("rtsp://", "http://"))
            else str(video_path)
        )
        cmd = [str(vlc_exe), target]

        if start_time_seconds is not None and start_time_seconds > 0:
            cmd.append(f"--start-time={int(start_time_seconds)}")

        try:
            subprocess.Popen(cmd, close_fds=True)
            logger.info(f"🎬 Vídeo aberto no VLC: {target}")
            return True
        except Exception as e:
            logger.error(f"Falha ao abrir vídeo no VLC: {e}")
            return False

    @classmethod
    def remux_to_clean_mp4(cls, input_path: str | Path, output_path: str | Path | None = None) -> Path | None:
        """Usa o motor do VLC para reempacotar (remux) streams .265/H.265 em um contêiner .mp4 limpo sem perda de qualidade."""
        vlc_exe = cls.get_vlc_executable()
        if not vlc_exe:
            return None

        inp = Path(input_path).resolve()
        if not inp.exists():
            return None

        if output_path is None:
            out = inp.with_name(f"{inp.stem}_vlc_clean.mp4")
        else:
            out = Path(output_path).resolve()

        out.parent.mkdir(parents=True, exist_ok=True)
        # Remove saída antiga: antes um arquivo de uma execução anterior "passava" na verificação
        # de tamanho mesmo quando o remux atual falhava
        try:
            out.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"Não foi possível remover saída anterior {out}: {exc}")
            return None

        cmd = [str(vlc_exe), "-I", "dummy", str(inp), f"--sout=#std{{access=file,mux=mp4,dst='{out}'}}", "vlc://quit"]

        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600)  # 120s era pouco p/ gravações longas
            if proc.returncode != 0:
                logger.warning(f"VLC retornou código {proc.returncode} no remux de {inp.name}")
            if out.exists() and out.stat().st_size > 1000:
                logger.info(f"✅ VLC remux concluído: {out}")
                return out
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Remux do VLC excedeu o tempo limite: {inp.name}")
        except Exception as e:
            logger.error(f"❌ Erro no remux do VLC: {e}")
        try:
            out.unlink(missing_ok=True)  # não deixa arquivo parcial
        except OSError:
            pass

        return None
