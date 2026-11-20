"""Widget visual de Drag & Drop para soltar arquivos de vídeo na aplicação."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import customtkinter as ctk

from src.presentation.desktop.theme import (
    BG_CARD,
    BG_CARD_HOVER,
    BORDER_COLOR,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    FONT_BODY_BOLD,
    FONT_SMALL,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

SUPPORTED_EXTS = {".mp4", ".265", ".h265", ".hevc", ".avi", ".mkv", ".mov"}


class DropTarget(ctk.CTkFrame):
    """Área visual destacada com suporte a Drag & Drop e clique para seleção de vídeo."""

    def __init__(
        self,
        master,
        on_file_selected: Callable[[str], None],
        height: int = 140,
        **kwargs,
    ):
        super().__init__(
            master,
            height=height,
            corner_radius=12,
            fg_color=BG_CARD,
            border_width=2,
            border_color=BORDER_COLOR,
            **kwargs,
        )
        self.on_file_selected = on_file_selected
        self.selected_file: str | None = None

        self._build_ui()
        self._setup_dnd()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        self.icon_label = ctk.CTkLabel(
            self,
            text="📥",
            font=("Segoe UI", 32),
            text_color=COLOR_PRIMARY,
        )
        self.icon_label.pack(pady=(15, 2))

        self.title_label = ctk.CTkLabel(
            self,
            text="Arraste e solte o arquivo de vídeo aqui",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY,
        )
        self.title_label.pack(pady=1)

        self.subtitle_label = ctk.CTkLabel(
            self,
            text="Formatos suportados: .mp4, .265 (HEVC), .avi, .mkv  |  ou clique para procurar",
            font=FONT_SMALL,
            text_color=TEXT_MUTED,
        )
        self.subtitle_label.pack(pady=(1, 10))

        # Tornar toda a área clicável
        for widget in (self, self.icon_label, self.title_label, self.subtitle_label):
            widget.bind("<Button-1>", lambda e: self._on_click_browse())
            widget.bind("<Enter>", lambda e: self._on_hover(True))
            widget.bind("<Leave>", lambda e: self._on_hover(False))

    def _setup_dnd(self) -> None:
        """Tenta habilitar o DnD nativo se o TkinterDnD estiver presente."""
        try:
            from tkinterdnd2 import DND_FILES

            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<DropEnter>>", self._on_drag_enter)
            self.dnd_bind("<<DropLeave>>", self._on_drag_leave)
            self.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            # Se o ambiente não tiver TkinterDnD carregado na janela raiz, o clique continua 100% funcional
            pass

    def _on_hover(self, is_hover: bool) -> None:
        color = BG_CARD_HOVER if is_hover else BG_CARD
        border = COLOR_PRIMARY if is_hover else BORDER_COLOR
        self.configure(fg_color=color, border_color=border)

    def _on_drag_enter(self, event) -> None:
        self.configure(border_color=COLOR_PRIMARY, fg_color=BG_CARD_HOVER)

    def _on_drag_leave(self, event) -> None:
        self.configure(border_color=BORDER_COLOR, fg_color=BG_CARD)

    def _on_drop(self, event) -> None:
        self._on_drag_leave(event)
        raw_path = event.data.strip()
        # Limpar chaves ou aspas do Windows ({C:\path with spaces})
        if raw_path.startswith("{") and raw_path.endswith("}"):
            raw_path = raw_path[1:-1]
        elif raw_path.startswith('"') and raw_path.endswith('"'):
            raw_path = raw_path[1:-1]

        self.set_file(raw_path)

    def _on_click_browse(self) -> None:
        from tkinter import filedialog

        filetypes = [
            ("Arquivos de Vídeo", "*.mp4 *.265 *.h265 *.hevc *.avi *.mkv *.mov"),
            ("Todos os Arquivos", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            title="Selecione o vídeo para análise de EPI",
            filetypes=filetypes,
        )
        if chosen:
            self.set_file(chosen)

    def set_file(self, filepath: str) -> None:
        """Define o arquivo selecionado e dispara o callback."""
        path_obj = Path(filepath)
        if not path_obj.exists():
            return

        ext = path_obj.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            self.title_label.configure(
                text=f"⚠️ Extensão {ext} não é um vídeo suportado",
                text_color=COLOR_PRIMARY,
            )
            return

        self.selected_file = str(path_obj.resolve())
        size_mb = path_obj.stat().st_size / (1024 * 1024)

        self.icon_label.configure(text="🎬", text_color=COLOR_SUCCESS)
        self.title_label.configure(
            text=f"Vídeo selecionado: {path_obj.name}",
            text_color=COLOR_SUCCESS,
        )
        self.subtitle_label.configure(
            text=f"Tamanho: {size_mb:.1f} MB  |  Extensão: {ext.upper()}  |  Clique para alterar",
            text_color=TEXT_SECONDARY,
        )
        self.configure(border_color=COLOR_SUCCESS)

        self.on_file_selected(self.selected_file)

    def reset(self) -> None:
        """Restaura o estado inicial do componente."""
        self.selected_file = None
        self.icon_label.configure(text="📥", text_color=COLOR_PRIMARY)
        self.title_label.configure(
            text="Arraste e solte o arquivo de vídeo aqui",
            text_color=TEXT_PRIMARY,
        )
        self.subtitle_label.configure(
            text="Formatos suportados: .mp4, .265 (HEVC), .avi, .mkv  |  ou clique para procurar",
            text_color=TEXT_MUTED,
        )
        self.configure(border_color=BORDER_COLOR, fg_color=BG_CARD)
