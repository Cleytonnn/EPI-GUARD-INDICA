"""Testes unitários para o serviço de integração com o VLC."""

from pathlib import Path
from src.infrastructure.video.vlc_service import VLCService


def test_vlc_service_detection():
    """Testa se o serviço detecta o executável do VLC e o runtime libvlc."""
    exe_path = VLCService.get_vlc_executable()
    assert exe_path is not None
    assert exe_path.exists()
    assert "vlc.exe" in str(exe_path).lower()

    assert VLCService.is_available() is True
    assert VLCService.is_libvlc_available() is True


def test_vlc_embedded_player_initialization():
    """Testa se o widget EmbeddedVLCPlayer inicializa corretamente."""
    import customtkinter as ctk
    from src.presentation.desktop.widgets.vlc_player_canvas import EmbeddedVLCPlayer

    root = ctk.CTk()
    root.withdraw()

    player = EmbeddedVLCPlayer(root, width=320, height=240)
    assert player is not None
    assert player.video_frame is not None

    root.destroy()

