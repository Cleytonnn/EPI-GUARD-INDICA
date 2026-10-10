"""Testes unitários para a aplicação Desktop (CustomTkinter)."""

import pytest
from src.container import Container
from src.presentation.desktop.app import PPEComplianceDesktopApp


def test_desktop_app_initialization_and_view_switching():
    """Testa se a aplicação desktop inicializa e troca de views sem erros."""
    container = Container()
    app = PPEComplianceDesktopApp(container=container)
    app.withdraw()  # Não exibir janela durante testes automatizados
    app.update()

    # Garantir que todas as telas registradas são instanciadas
    expected_views = ["file_analysis", "batch_view", "live_view", "evidence_view"]
    for view_key in expected_views:
        assert view_key in app.views
        app._show_view(view_key)
        app.update()
        assert app._current_view is not None

    app.destroy()
