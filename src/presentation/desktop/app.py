"""Aplicação Desktop Principal para Monitoramento e Análise de EPI."""

from __future__ import annotations

import logging
import sys
from typing import Optional
import customtkinter as ctk

from src.container import Container
from src.presentation.desktop.theme import (
    BG_MAIN,
    BG_SIDEBAR,
    BORDER_COLOR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_HEADING,
    FONT_SMALL,
    FONT_SUBTITLE,
    FONT_TITLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from src.presentation.desktop.views.batch_view import BatchView
from src.presentation.desktop.views.evidence_view import EvidenceView
from src.presentation.desktop.views.file_analysis_view import FileAnalysisView
from src.presentation.desktop.views.live_view import LiveView
from src.presentation.desktop.views.training_review_view import TrainingReviewView

logger = logging.getLogger(__name__)

# Tentar integrar com o TkinterDnD para suporte nativo a Drag & Drop em toda a janela
try:
    from tkinterdnd2 import TkinterDnD
    BaseAppClass = type("BaseAppClass", (ctk.CTk, TkinterDnD.DnDWrapper), {})
except Exception:
    BaseAppClass = ctk.CTk


class PPEComplianceDesktopApp(BaseAppClass):
    """Janela principal do aplicativo Desktop de Detecção e Conformidade de EPI."""

    def __init__(self, container: Optional[Container] = None):
        super().__init__()
        if hasattr(self, "TkdndVersion") is False and "TkinterDnD" in globals():
            try:
                self.TkdndVersion = TkinterDnD._require(self)
            except Exception:
                pass

        self.container = container or Container()

        # Configurações do CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("🛡️ Sistema Corporativo de Detecção de EPI - Desktop")
        self.geometry("1240x780")
        self.minsize(1050, 680)
        self.configure(fg_color=BG_MAIN)

        self._current_view: Optional[ctk.CTkFrame] = None
        self._nav_buttons = {}

        self._build_layout()
        self._show_view("file_analysis")

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)  # Sidebar fixa
        self.grid_columnconfigure(1, weight=1)  # Conteúdo dinâmico
        self.grid_rowconfigure(0, weight=1)

        # 1. Barra Lateral (Sidebar)
        self.sidebar = ctk.CTkFrame(
            self,
            width=260,
            corner_radius=0,
            fg_color=BG_SIDEBAR,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # Logotipo / Cabeçalho da Sidebar
        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=20, pady=(20, 25))

        ctk.CTkLabel(
            logo_frame,
            text="🛡️ EPI GUARD",
            font=FONT_TITLE,
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")

        ctk.CTkLabel(
            logo_frame,
            text="Sistema de Conformidade em IA",
            font=FONT_SMALL,
            text_color=COLOR_PRIMARY,
        ).pack(anchor="w")

        # Botões de Navegação
        nav_items = [
            ("file_analysis", "🎬 Analisar Arquivo", "Solte vídeos .mp4 ou .265"),
            ("batch_view", "📂 Lote & Pastas", "Varredura [PLACA]>[DATA]"),
            ("live_view", "📹 Câmera ao Vivo", "Transmissão RTSP / Webcam"),
            ("evidence_view", "🚨 Alertas & Evidências", "Fotos salvas e relatório CSV"),
            ("training_review", "🧠 Revisão p/ Treino", "Confirme frames para retreinar a IA"),
        ]

        for view_key, title, subtitle in nav_items:
            btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            btn_frame.pack(fill="x", padx=12, pady=4)

            btn = ctk.CTkButton(
                btn_frame,
                text=title,
                font=FONT_BODY_BOLD,
                fg_color="transparent",
                hover_color="#1e293b",
                text_color=TEXT_SECONDARY,
                anchor="w",
                height=42,
                corner_radius=8,
                command=lambda k=view_key: self._show_view(k),
            )
            btn.pack(fill="x")
            self._nav_buttons[view_key] = btn

        # Rodapé da Sidebar (Informações do Hardware & Modelo)
        footer_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="#080c10",
            corner_radius=10,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        footer_frame.pack(side="bottom", fill="x", padx=12, pady=15)

        ctk.CTkLabel(
            footer_frame,
            text="STATUS DO SISTEMA",
            font=("Segoe UI", 9, "bold"),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=12, pady=(10, 4))

        device_text = f"🟢 GPU: {self.container.detector.device.upper()}"
        ctk.CTkLabel(
            footer_frame,
            text=device_text,
            font=FONT_SMALL,
            text_color=COLOR_SUCCESS,
        ).pack(anchor="w", padx=12, pady=1)

        model_name = self.container.settings.detection.model_path
        ctk.CTkLabel(
            footer_frame,
            text=f"🤖 Modelo: {model_name}",
            font=FONT_SMALL,
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", padx=12, pady=1)

        ctk.CTkLabel(
            footer_frame,
            text="Versão 2.0 (Clean Architecture)",
            font=("Segoe UI", 8),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", padx=12, pady=(4, 10))

        # 2. Container de Conteúdo Principal
        self.content_container = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        self.content_container.grid(row=0, column=1, sticky="nsew")
        self.content_container.grid_columnconfigure(0, weight=1)
        self.content_container.grid_rowconfigure(0, weight=1)

        # Inicialização preguiçosa das telas
        self.views = {
            "file_analysis": lambda: FileAnalysisView(self.content_container, self.container),
            "batch_view": lambda: BatchView(self.content_container, self.container),
            "live_view": lambda: LiveView(self.content_container, self.container),
            "evidence_view": lambda: EvidenceView(self.content_container, self.container),
            "training_review": lambda: TrainingReviewView(self.content_container, self.container),
        }
        self._cached_views = {}

    def _show_view(self, view_key: str) -> None:
        """Alterna a tela visível na aplicação."""
        if self._current_view is not None:
            self._current_view.grid_forget()

        # Atualizar destaque visual dos botões na sidebar
        for k, btn in self._nav_buttons.items():
            if k == view_key:
                btn.configure(fg_color="#1e293b", text_color=TEXT_PRIMARY)
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_SECONDARY)

        if view_key not in self._cached_views:
            self._cached_views[view_key] = self.views[view_key]()

        self._current_view = self._cached_views[view_key]
        self._current_view.grid(row=0, column=0, sticky="nsew")


def run_desktop_app(container: Optional[Container] = None) -> None:
    """Inicia a aplicação desktop em loop de eventos da interface gráfica."""
    app = PPEComplianceDesktopApp(container=container)
    app.mainloop()


if __name__ == "__main__":
    run_desktop_app()
