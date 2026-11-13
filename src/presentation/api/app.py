"""Fábrica da aplicação FastAPI para a API REST."""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.container import Container
from src.presentation.api.routes import router

logger = logging.getLogger(__name__)

# Container singleton global para a API
app_container: Optional[Container] = None


def create_app(container: Optional[Container] = None) -> FastAPI:
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
    cors_origins = app_container.settings.api.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
