"""Testes unitários para o serviço de integração com o VLC.

Dependem do VLC instalado e de um runtime Tk funcional; quando o ambiente
não os provê (ex.: CI headless), os testes são ignorados em vez de falhar.
"""

import pytest

from src.infrastructure.video.vlc_service import VLCService


@pytest.mark.skipif(not VLCService.is_available(), reason="VLC não instalado neste ambiente")
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
    import tkinter

    import customtkinter as ctk

    from src.presentation.desktop.widgets.vlc_player_canvas import EmbeddedVLCPlayer

    try:
        root = ctk.CTk()
    except tkinter.TclError as exc:
        pytest.skip(f"Runtime Tk indisponível: {exc}")

    root.withdraw()
    try:
        player = EmbeddedVLCPlayer(root, width=320, height=240)
        assert player is not None
        assert player.video_frame is not None
    finally:
        root.destroy()
