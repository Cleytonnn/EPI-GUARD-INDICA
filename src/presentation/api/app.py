"""Fábrica da aplicação FastAPI para a API REST."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.container import Container
from src.presentation.api.routes import router

logger = logging.getLogger(__name__)

# Container singleton global para a API
app_container: Container | None = None


def create_app(container: Container | None = None) -> FastAPI:
    """Cria e configura a instância do FastAPI com injeção de dependências."""
    global app_container
    app_container = container or Container()

    app = FastAPI(
        title="🛡️ Sistema de Detecção de EPI - API REST",
        description=(
            "API corporativa para inferência e compliance de Equipamentos de Proteção Individual "
            "em tempo real. CRIADO POR CLEYTON BAUMGRATZ."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configuração de CORS
    # "*" + allow_credentials=True é inválido pela especificação CORS (navegadores rejeitam)
    # e expõe a API a qualquer site. Credenciais só são liberadas com origens explícitas.
    cors_origins = list(app_container.settings.api.cors_origins or [])
    wildcard = "*" in cors_origins
    if wildcard:
        logger.warning("⚠️ CORS liberado para qualquer origem ('*'). Defina api.cors_origins em produção.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not wildcard,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )
    if not getattr(app_container.settings.api, "api_key", None):
        logger.warning("⚠️ API sem autenticação (defina API_KEY no .env para exigir o header X-API-Key).")

    # Registro de rotas (adiciona as rotas diretamente para compatibilidade de inspeção e OpenAPI limpo)
    for r in router.routes:
        app.routes.append(r)

    @app.get("/", tags=["Geral"])
    def root():
        return {
            "message": "Sistema de Detecção de EPI - API REST ativa",
            "author": "CRIADO POR CLEYTON BAUMGRATZ",
            "docs": "/docs",
            "health": "/health",
            "version": "2.0.0",
        }

    return app
