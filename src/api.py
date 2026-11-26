"""Ponto de entrada da API REST com suporte retrocompatível.

Uso:
    python -m src.api
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse

import uvicorn

from src.container import Container
from src.presentation.api.app import create_app

# Instância padrão do FastAPI exportada no módulo
container = Container()
app = create_app(container)


def main():
    parser = argparse.ArgumentParser(description="Servidor de API REST para Detecção de EPI")
    parser.add_argument("--host", default=container.settings.api.host, help="Host para bind")
    parser.add_argument("--port", type=int, default=container.settings.api.port, help="Porta do servidor")
    parser.add_argument("--reload", action="store_true", help="Recarregar em alterações (desenvolvimento)")
    args = parser.parse_args()

    # Sem --reload, passa o objeto `app` já criado. Antes era sempre a string "src.api:app",
    # o que fazia o uvicorn reimportar o módulo e carregar os modelos de IA DUAS vezes.
    uvicorn.run(
        "src.api:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
