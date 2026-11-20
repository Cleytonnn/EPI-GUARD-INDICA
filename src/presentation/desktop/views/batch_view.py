"""Tela de Processamento em Lote e Monitoramento Contínuo de Pastas (Frota)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from src.container import Container
from src.presentation.desktop.theme import (
    BG_CARD,
    BG_INPUT,
    BORDER_COLOR,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_SMALL,
    FONT_TITLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class BatchView(ctk.CTkFrame):
    """View para varredura recursiva de diretórios por placa e data, e Watch Mode."""

    def __init__(self, master, container: Container, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.container = container

        self.is_running: bool = False
        self.stop_requested: bool = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Cabeçalho
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="📂 Processamento em Lote & Monitoramento de Pastas",
            font=FONT_TITLE,
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Varre diretórios estruturados por [PLACA] > [DATA] > [ARQUIVO] (.mp4, .265) ou monitora descargas contínuas.",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(2, 0))

        # 2. Configurações da Varredura
        config_card = ctk.CTkFrame(
            self,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        config_card.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        config_card.grid_columnconfigure(1, weight=1)

        # Linha 1: Seleção de Pasta
        ctk.CTkLabel(config_card, text="Diretório Raiz:", font=FONT_BODY_BOLD, text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=(15, 10), pady=(15, 5)
        )

        default_dir = str(Path("gravacoes").resolve()) if Path("gravacoes").exists() else str(Path.cwd())
        self.dir_entry = ctk.CTkEntry(
            config_card,
            font=FONT_BODY,
            height=32,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
        )
        self.dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(15, 5))
        self.dir_entry.insert(0, default_dir)

        self.browse_btn = ctk.CTkButton(
            config_card,
            text="Procurar...",
            font=FONT_BODY,
            width=100,
            height=32,
            command=self._on_browse_dir,
        )
        self.browse_btn.grid(row=0, column=2, padx=(0, 15), pady=(15, 5))

        # Linha 2: Opções
        opts_frame = ctk.CTkFrame(config_card, fg_color="transparent")
        opts_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=15, pady=(5, 10))

        self.force_check = ctk.CTkCheckBox(
            opts_frame,
            text="Forçar reprocessamento de vídeos já vistos (--force)",
            font=FONT_BODY,
        )
        self.force_check.pack(side="left", padx=(0, 20))

        self.watch_switch = ctk.CTkSwitch(
            opts_frame,
            text="Monitoramento Contínuo (Watch Mode)",
            font=FONT_BODY_BOLD,
        )
        self.watch_switch.pack(side="left", padx=(0, 10))

        # Linha 3: Botões de Ação
        actions = ctk.CTkFrame(config_card, fg_color="transparent")
        actions.grid(row=2, column=0, columnspan=3, sticky="ew", padx=15, pady=(5, 15))

        self.start_btn = ctk.CTkButton(
            actions,
            text="🚀 Iniciar Varredura em Lote",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_PRIMARY,
            hover_color="#1565c0",
            height=34,
            command=self._on_start,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.stop_btn = ctk.CTkButton(
            actions,
            text="⏹️ Parar Operação",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_DANGER,
            hover_color="#b91c1c",
            height=34,
            state="disabled",
            command=self._on_stop,
        )
        self.stop_btn.pack(side="left")

        # 3. Terminal de Log e Andamento
        log_card = ctk.CTkFrame(
            self,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        log_card.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 15))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        log_header = ctk.CTkFrame(log_card, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        ctk.CTkLabel(
            log_header,
            text="📜 Log de Execução e Auditoria",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        self.status_lbl = ctk.CTkLabel(
            log_header,
            text="OCIOSO",
            font=FONT_SMALL,
            fg_color="#334155",
            corner_radius=6,
            padx=8,
            pady=2,
        )
        self.status_lbl.pack(side="right")

        self.log_textbox = ctk.CTkTextbox(
            log_card,
            font=("Consolas", 11),
            fg_color=BG_INPUT,
            border_width=1,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
        )
        self.log_textbox.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        self._append_log("Sistema pronto. Selecione o diretório e clique em Iniciar.")

    def _append_log(self, text: str) -> None:
        t_str = time.strftime("%H:%M:%S")
        self.log_textbox.insert("end", f"[{t_str}] {text}\n")
        self.log_textbox.see("end")

    def _on_browse_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecione o diretório com as gravações da frota")
        if folder:
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, folder)

    def _on_start(self) -> None:
        target_dir = self.dir_entry.get().strip()
        if not target_dir or not Path(target_dir).exists():
            self._append_log(f"❌ Diretório inválido ou não encontrado: {target_dir}")
            return

        self.is_running = True
        self.stop_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.browse_btn.configure(state="disabled")

        is_watch = bool(self.watch_switch.get())
        force = bool(self.force_check.get())

        mode_desc = "Watch Mode (Monitoramento Contínuo)" if is_watch else "Varredura em Lote Única"
        self.status_lbl.configure(text="EXECUTANDO", fg_color=COLOR_WARNING)
        self._append_log(f"Iniciando {mode_desc} em: {target_dir}")

        threading.Thread(
            target=self._run_batch_thread,
            args=(target_dir, force, is_watch),
            daemon=True,
        ).start()

    def _on_stop(self) -> None:
        self.stop_requested = True
        self.container.watch_directory_uc.stop()
        self._append_log("Solicitação de parada enviada. Aguardando finalização...")

    def _run_batch_thread(self, target_dir: str, force: bool, is_watch: bool) -> None:
        try:
            if is_watch:
                self.after(
                    0,
                    self._append_log,
                    "👁️ Modo Watcher ativo. Monitorando novos arquivos de vídeo gravados...",
                )
                self.container.watch_directory_uc.execute(
                    input_dir=target_dir,
                    interval_seconds=10,
                )
            else:
                results = self.container.process_batch_uc.execute(
                    input_dir=target_dir,
                    force_reprocess=force,
                )
                total_viols = sum(r.violations_count for r in results)
                self.after(
                    0,
                    self._append_log,
                    f"✅ Processamento em lote concluído! Total de vídeos: {len(results)} | Total de Infrações: {total_viols}",
                )

            self.after(0, self._on_finish)
        except Exception as e:
            self.after(0, self._append_log, f"❌ Erro na execução: {e}")
            self.after(0, self._on_finish)

    def _on_finish(self) -> None:
        self.is_running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.browse_btn.configure(state="normal")
        self.status_lbl.configure(text="CONCLUÍDO", fg_color=COLOR_SUCCESS)
