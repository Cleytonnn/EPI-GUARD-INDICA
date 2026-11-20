"""Script de inicialização rápida da Interface Desktop de Detecção de EPI.

Permite iniciar a aplicação diretamente com dois cliques no Windows ou executando:
    python run_desktop.py
"""

from src.container import Container
from src.presentation.desktop import run_desktop_app

if __name__ == "__main__":
    container = Container()
    run_desktop_app(container=container)
