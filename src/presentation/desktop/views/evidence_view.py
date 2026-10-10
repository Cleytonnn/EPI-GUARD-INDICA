"""Tela de Evidências, Galeria de Fotos e Relatório CSV de Infrações."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional
from PIL import Image, ImageTk
import customtkinter as ctk

from src.container import Container
from src.presentation.desktop.theme import (
    BG_CARD,
    BG_INPUT,
    BORDER_COLOR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_SMALL,
    FONT_TITLE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class EvidenceView(ctk.CTkFrame):
    """View para exibição de evidências salvas e abertura dos relatórios da empresa."""

    def __init__(self, master, container: Container, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.container = container
        self.selected_image_path: Optional[str] = None
        self._current_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._refresh_data()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Cabeçalho
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="🚨 Galeria de Evidências & Auditoria",
            font=FONT_TITLE,
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Visualize as fotos de infração gravadas com anotações e tarja de auditoria, ou exporte a planilha CSV.",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(2, 0))

        # 2. Barra de Ferramentas
        toolbar = ctk.CTkFrame(
            self,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        toolbar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        self.refresh_btn = ctk.CTkButton(
            toolbar,
            text="🔄 Atualizar Lista",
            font=FONT_BODY,
            width=120,
            command=self._refresh_data,
        )
        self.refresh_btn.pack(side="left", padx=15, pady=10)

        self.open_folder_btn = ctk.CTkButton(
            toolbar,
            text="📂 Abrir Pasta 'alertas/' no Windows",
            font=FONT_BODY,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_alerts_folder,
        )
        self.open_folder_btn.pack(side="left", padx=(0, 10), pady=10)

        self.open_csv_btn = ctk.CTkButton(
            toolbar,
            text="📊 Abrir Planilha CSV no Excel",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_SUCCESS,
            hover_color="#059669",
            command=self._open_csv_report,
        )
        self.open_csv_btn.pack(side="left", padx=(0, 15), pady=10)

        # 3. Conteúdo (Lista de Fotos + Visualizador em Detalhe)
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 15))
        content.grid_columnconfigure(0, weight=2)
        content.grid_columnconfigure(1, weight=3)
        content.grid_rowconfigure(0, weight=1)

        # Coluna Esquerda: Lista de arquivos
        list_card = ctk.CTkFrame(
            content,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        list_card.grid_columnconfigure(0, weight=1)
        list_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            list_card,
            text="📸 Imagens de Evidências Encontradas",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))

        self.images_scroll = ctk.CTkScrollableFrame(
            list_card,
            fg_color=BG_INPUT,
            corner_radius=8,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        self.images_scroll.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Coluna Direita: Visualizador da Imagem Selecionada
        preview_card = ctk.CTkFrame(
            content,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        preview_card.grid(row=0, column=1, sticky="nsew")
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(1, weight=1)

        self.preview_header = ctk.CTkLabel(
            preview_card,
            text="Selecione uma imagem para visualizar",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY,
        )
        self.preview_header.grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))

        self.preview_canvas = ctk.CTkCanvas(
            preview_card,
            bg=BG_INPUT,
            highlightthickness=0,
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

    def _refresh_data(self) -> None:
        for child in self.images_scroll.winfo_children():
            child.destroy()

        alerts_dir = Path(self.container.settings.alerts.save_dir).resolve()
        if not alerts_dir.exists():
            lbl = ctk.CTkLabel(self.images_scroll, text="Pasta 'alertas/' ainda não criada.", text_color=TEXT_MUTED)
            lbl.pack(pady=20)
            return

        image_files = list(alerts_dir.rglob("*.jpg")) + list(alerts_dir.rglob("*.png"))
        image_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if not image_files:
            lbl = ctk.CTkLabel(self.images_scroll, text="Nenhuma evidência registrada ainda.", text_color=TEXT_MUTED)
            lbl.pack(pady=20)
            return

        for img_path in image_files[:100]:
            rel_name = img_path.relative_to(alerts_dir)
            btn = ctk.CTkButton(
                self.images_scroll,
                text=f"📷 {rel_name}",
                font=FONT_SMALL,
                fg_color=BG_CARD,
                hover_color="#334155",
                anchor="w",
                height=32,
                command=lambda p=str(img_path): self._display_image(p),
            )
            btn.pack(fill="x", pady=2, padx=2)

    def _display_image(self, img_path_str: str) -> None:
        self.selected_image_path = img_path_str
        path_obj = Path(img_path_str)
        self.preview_header.configure(text=f"Evidência: {path_obj.name}")

        try:
            pil_img = Image.open(img_path_str)
            w, h = pil_img.size

            canvas_w = max(self.preview_canvas.winfo_width(), 400)
            canvas_h = max(self.preview_canvas.winfo_height(), 300)

            scale = min(canvas_w / w, canvas_h / h)
            new_w = max(int(w * scale), 1)
            new_h = max(int(h * scale), 1)

            resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self._current_photo = ImageTk.PhotoImage(resized)

            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                canvas_w // 2, canvas_h // 2, image=self._current_photo, anchor="center"
            )
        except Exception as e:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                200, 150, text=f"Erro ao carregar imagem: {e}", fill="#ef4444"
            )

    def _open_alerts_folder(self) -> None:
        p = Path(self.container.settings.alerts.save_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(p)

    def _open_csv_report(self) -> None:
        p = Path(self.container.settings.batch.output_csv).resolve()
        if p.exists():
            os.startfile(p)
        else:
            self._open_alerts_folder()
