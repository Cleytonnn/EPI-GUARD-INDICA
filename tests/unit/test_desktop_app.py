"""Testes unitários para a aplicação Desktop (CustomTkinter).

Ignorados automaticamente quando o Python não tem Tkinter ou não há display (CI headless).
"""

import pytest

tkinter = pytest.importorskip("tkinter", reason="Tkinter indisponível neste Python")
pytest.importorskip("customtkinter")

from src.container import Container  # noqa: E402
from src.presentation.desktop.app import PPEComplianceDesktopApp  # noqa: E402


def test_desktop_app_initialization_and_view_switching():
    """Testa se a aplicação desktop inicializa e troca de views sem erros."""
    container = Container()
    try:
        app = PPEComplianceDesktopApp(container=container)
    except tkinter.TclError as exc:
        pytest.skip(f"Sem display gráfico disponível: {exc}")
    app.withdraw()  # Não exibir janela durante testes automatizados
    app.update()

    # Garantir que todas as telas registradas são instanciadas
    expected_views = ["file_analysis", "batch_view", "live_view", "evidence_view", "training_review"]
    for view_key in expected_views:
        assert view_key in app.views
        app._show_view(view_key)
        app.update()
        assert app._current_view is not None

    app.destroy()
