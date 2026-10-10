"""Tela de Análise Direta de Vídeo (Upload, Drag & Drop e Processamento Frame a Frame)."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any, List, Optional
import customtkinter as ctk

from src.container import Container
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.use_cases.process_video_file import VideoProcessingResult
from src.presentation.desktop.theme import (
    BG_CARD,
    BG_INPUT,
    BG_MAIN,
    BORDER_COLOR,
    COLOR_ACCENT,
    COLOR_DANGER,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_WARNING,
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
from src.presentation.desktop.widgets.drop_target import DropTarget
from src.presentation.desktop.widgets.video_canvas import VideoCanvas


class FileAnalysisView(ctk.CTkFrame):
    """View responsável pela análise de vídeos individuais soltos ou selecionados."""

    def __init__(self, master, container: Container, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.container = container

        self.selected_file: Optional[str] = None
        self.is_processing: bool = False
        self.stop_requested: bool = False
        self._current_metadata: Optional[VehicleMetadata] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Cabeçalho da View
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 10))

        title = ctk.CTkLabel(
            header_frame,
            text="🎬 Análise Direta de Gravação (.mp4 / .265)",
            font=FONT_TITLE,
            text_color=TEXT_PRIMARY,
        )
        title.pack(anchor="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Arraste o vídeo para a área abaixo ou selecione no seu computador. A IA analisará o uso de EPI frame a frame.",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        )
        subtitle.pack(anchor="w", pady=(2, 0))

        # 2. Área de Drag & Drop e Configurações de Metadados
        top_section = ctk.CTkFrame(self, fg_color="transparent")
        top_section.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        top_section.grid_columnconfigure(0, weight=3)
        top_section.grid_columnconfigure(1, weight=2)

        self.drop_target = DropTarget(
            top_section,
            on_file_selected=self._on_file_chosen,
            height=130,
        )
        self.drop_target.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Painel lateral de metadados detectados
        meta_panel = ctk.CTkFrame(
            top_section,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        meta_panel.grid(row=0, column=1, sticky="nsew")
        meta_panel.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            meta_panel,
            text="🏷️ Informações da Gravação",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 5))

        # Placa do veículo
        ctk.CTkLabel(meta_panel, text="Placa:", font=FONT_SMALL, text_color=TEXT_MUTED).grid(
            row=1, column=0, sticky="w", padx=(15, 5), pady=2
        )
        self.plate_entry = ctk.CTkEntry(
            meta_panel,
            font=FONT_BODY,
            height=28,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
        )
        self.plate_entry.grid(row=1, column=1, sticky="ew", padx=(0, 15), pady=2)
        self.plate_entry.insert(0, "AUTO_DETECT")

        # Data
        ctk.CTkLabel(meta_panel, text="Data:", font=FONT_SMALL, text_color=TEXT_MUTED).grid(
            row=2, column=0, sticky="w", padx=(15, 5), pady=2
        )
        self.date_entry = ctk.CTkEntry(
            meta_panel,
            font=FONT_BODY,
            height=28,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
        )
        self.date_entry.grid(row=2, column=1, sticky="ew", padx=(0, 15), pady=2)
        self.date_entry.insert(0, "AUTO_DETECT")

        # Sensibilidade / Limiar de Confiança da IA
        ctk.CTkLabel(meta_panel, text="Confiança:", font=FONT_SMALL, text_color=TEXT_MUTED).grid(
            row=3, column=0, sticky="w", padx=(15, 5), pady=2
        )
        slider_frame = ctk.CTkFrame(meta_panel, fg_color="transparent")
        slider_frame.grid(row=3, column=1, sticky="ew", padx=(0, 15), pady=2)
        slider_frame.grid_columnconfigure(0, weight=1)

        self.conf_slider = ctk.CTkSlider(
            slider_frame,
            from_=0.10,
            to=0.70,
            number_of_steps=12,
            height=14,
            command=self._on_conf_slider_change,
        )
        self.conf_slider.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.conf_slider.set(0.20)

        self.conf_lbl = ctk.CTkLabel(
            slider_frame,
            text="20%",
            font=FONT_SMALL,
            text_color=COLOR_ACCENT,
            width=36,
        )
        self.conf_lbl.grid(row=0, column=1)

        # Botões de Ação
        actions_frame = ctk.CTkFrame(meta_panel, fg_color="transparent")
        actions_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=15, pady=(8, 10))

        self.start_btn = ctk.CTkButton(
            actions_frame,
            text="▶️ Iniciar Análise",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_PRIMARY,
            hover_color="#1565c0",
            height=32,
            command=self._on_start_analysis,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.stop_btn = ctk.CTkButton(
            actions_frame,
            text="⏹️ Cancelar",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_DANGER,
            hover_color="#b91c1c",
            height=32,
            state="disabled",
            command=self._on_stop_analysis,
        )
        self.stop_btn.pack(side="left", padx=(5, 0))

        # 3. Barra de Progresso
        progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        progress_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 5))
        progress_frame.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(
            progress_frame,
            fg_color=BG_INPUT,
            progress_color=COLOR_PRIMARY,
            height=10,
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        self.progress_bar.set(0.0)

        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Aguardando seleção de vídeo para processar.",
            font=FONT_SMALL,
            text_color=TEXT_MUTED,
        )
        self.progress_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

        # 4. Área Principal (Player de Vídeo + Painel de Infrações)
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.grid(row=3, column=0, sticky="nsew", padx=20, pady=(5, 15))
        main_content.grid_columnconfigure(0, weight=3)
        main_content.grid_columnconfigure(1, weight=2)
        main_content.grid_rowconfigure(0, weight=1)

        # Player
        player_container = ctk.CTkFrame(
            main_content,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        player_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        player_container.grid_columnconfigure(0, weight=1)
        player_container.grid_rowconfigure(1, weight=1)

        player_header = ctk.CTkFrame(player_container, fg_color="transparent")
        player_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))
        ctk.CTkLabel(
            player_header,
            text="📺 Reprodução com Detecções em Tempo Real",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        self.fps_badge = ctk.CTkLabel(
            player_header,
            text="0.0 FPS",
            font=FONT_SMALL,
            text_color=COLOR_ACCENT,
        )
        self.fps_badge.pack(side="right")

        self.video_canvas = VideoCanvas(player_container, width=560, height=340)
        self.video_canvas.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Painel de Resultados & Infrações
        results_container = ctk.CTkFrame(
            main_content,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        results_container.grid(row=0, column=1, sticky="nsew")
        results_container.grid_columnconfigure(0, weight=1)
        results_container.grid_rowconfigure(2, weight=1)

        # Cabeçalho dos Resultados
        results_header = ctk.CTkFrame(results_container, fg_color="transparent")
        results_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        ctk.CTkLabel(
            results_header,
            text="🚨 Infrações Identificadas",
            font=FONT_BODY_BOLD,
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        self.status_badge = ctk.CTkLabel(
            results_header,
            text="PRONTO",
            font=FONT_SMALL,
            fg_color="#334155",
            text_color=TEXT_PRIMARY,
            corner_radius=6,
            padx=8,
            pady=2,
        )
        self.status_badge.pack(side="right")

        # Sumário rápido
        self.summary_label = ctk.CTkLabel(
            results_container,
            text="Nenhuma infração registrada no momento.",
            font=FONT_SMALL,
            text_color=TEXT_MUTED,
            justify="left",
            anchor="w",
        )
        self.summary_label.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 5))

        # Lista com scroll das infrações
        self.violations_scroll = ctk.CTkScrollableFrame(
            results_container,
            fg_color=BG_INPUT,
            corner_radius=8,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        self.violations_scroll.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))

        # Botão para abrir pasta de evidências
        self.open_folder_btn = ctk.CTkButton(
            results_container,
            text="📂 Abrir Pasta de Evidências Salvas",
            font=FONT_BODY,
            fg_color="#334155",
            hover_color="#475569",
            height=30,
            command=self._on_open_evidence_folder,
        )
        self.open_folder_btn.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))

    def _on_file_chosen(self, filepath: str) -> None:
        """Executado quando o usuário solta ou seleciona um arquivo."""
        self.selected_file = filepath
        path_obj = Path(filepath)

        # Detectar metadados automáticos
        meta = VehicleMetadata.from_path(path_obj)
        self._current_metadata = meta

        self.plate_entry.delete(0, "end")
        self.plate_entry.insert(0, meta.plate)

        self.date_entry.delete(0, "end")
        self.date_entry.insert(0, meta.record_date)

        self.progress_label.configure(
            text=f"Pronto para analisar: {path_obj.name} (Tamanho: {path_obj.stat().st_size / (1024*1024):.1f} MB)",
            text_color=COLOR_SUCCESS,
        )
        self._clear_violations_list()
        self.video_canvas.show_placeholder("Clique em 'Iniciar Análise' para começar")

    def _clear_violations_list(self) -> None:
        for child in self.violations_scroll.winfo_children():
            child.destroy()
        self.status_badge.configure(text="AGUARDANDO", fg_color="#334155")
        self.summary_label.configure(text="Nenhuma infração registrada no momento.")

    def _on_conf_slider_change(self, val: float) -> None:
        pct = int(val * 100)
        self.conf_lbl.configure(text=f"{pct}%")
        self.container.detector.confidence = float(val)

    def _on_start_analysis(self) -> None:
        if not self.selected_file or not Path(self.selected_file).exists():
            self.progress_label.configure(
                text="⚠️ Por favor, solte ou selecione um arquivo de vídeo primeiro.",
                text_color=COLOR_WARNING,
            )
            return

        if self.is_processing:
            return

        # Aplicar limiar de confiança selecionado
        selected_conf = float(self.conf_slider.get())
        self.container.detector.confidence = selected_conf

        self.is_processing = True
        self.stop_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_bar.set(0.0)
        self._clear_violations_list()

        self.status_badge.configure(text="ANALISANDO...", fg_color=COLOR_WARNING)
        self.progress_label.configure(text="Inicializando decodificador e carregando IA...", text_color=TEXT_PRIMARY)

        # Disparar em thread de background para não travar a interface
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()

    def _on_stop_analysis(self) -> None:
        self.stop_requested = True
        self.progress_label.configure(text="Cancelando análise...", text_color=COLOR_DANGER)

    def _run_analysis_thread(self) -> None:
        video_path = self.selected_file
        t_start = time.time()
        frames_seen = 0
        total_violations_found = []

        try:
            # Override da placa se usuário editou
            custom_plate = self.plate_entry.get().strip()
            custom_date = self.date_entry.get().strip()

            reader = self.container.video_reader_factory(video_path)
            if not reader.is_opened():
                self.after(
                    0,
                    self._on_analysis_failed,
                    "Não foi possível abrir o arquivo. O arquivo pode estar corrompido ou vazio.",
                )
                return

            total_frames = reader.total_frames
            fps = reader.fps or 25.0

            # Zerar motor de compliance
            self.container.compliance_service.reset()

            base_meta = VehicleMetadata.from_path(video_path)
            plate_val = custom_plate if (custom_plate and custom_plate != "AUTO_DETECT") else base_meta.plate
            date_val = custom_date if (custom_date and custom_date != "AUTO_DETECT") else base_meta.record_date

            metadata = VehicleMetadata(
                plate=plate_val,
                record_date=date_val,
                video_name=Path(video_path).name,
            )

            skip_frames = self.container.settings.detection.skip_frames or 2
            consecutive_target = self.container.settings.compliance.min_violation_frames

            while not self.stop_requested:
                ret, frame = reader.read()
                if not ret or frame is None:
                    break

                frames_seen += 1
                if frames_seen % skip_frames != 0:
                    continue

                # Inferência YOLO
                detections = self.container.detector.detect(frame, skip_frames=1)

                # Regra de compliance
                timestamp_sec = frames_seen / fps
                current_meta = metadata.with_time(timestamp_sec, frames_seen)
                confirmed_violations = self.container.compliance_service.check(
                    detections=detections,
                    metadata=current_meta,
                )

                # Desenhar anotações e tarja
                from src.presentation.cli.visualizer import draw_detections, draw_vehicle_metadata

                annotated_frame = draw_detections(frame, detections)
                annotated_frame = draw_vehicle_metadata(
                    annotated_frame,
                    plate=current_meta.plate,
                    record_date=current_meta.record_date,
                    video_name=current_meta.video_name,
                    video_time=current_meta.video_time_str,
                )

                # Se houver violação confirmada, salvar imagem e registrar
                if confirmed_violations:
                    for v in confirmed_violations:
                        total_violations_found.append(v)
                        # Salvar imagem
                        saved_path = self.container.alert_storage.save_image(annotated_frame, v)
                        # Salvar relatório CSV
                        self.container.report_repo.log_violation(v, image_path=saved_path)
                        # Notificar
                        if self.container.notifier:
                            self.container.notifier.notify(v, image_path=saved_path)
                        # Atualizar UI
                        self.after(0, self._add_violation_item, v, timestamp_sec)

                progress = min(frames_seen / total_frames, 1.0) if total_frames > 0 else 0.5
                elapsed = time.time() - t_start
                current_fps = frames_seen / elapsed if elapsed > 0 else 0

                self.after(0, self._update_player, annotated_frame, progress, frames_seen, total_frames, current_fps)

            reader.release()

            if self.stop_requested:
                self.after(0, self._on_analysis_stopped)
            else:
                elapsed = time.time() - t_start
                self.after(0, self._on_analysis_finished, len(total_violations_found), elapsed, frames_seen)

        except Exception as e:
            self.after(0, self._on_analysis_failed, str(e))

    def _update_player(
        self,
        frame,
        progress: float,
        frame_idx: int,
        total_frames: int,
        current_fps: float,
    ) -> None:
        self.video_canvas.update_frame(frame)
        self.progress_bar.set(progress)
        self.fps_badge.configure(text=f"{current_fps:.1f} FPS")
        txt = f"Processando: Frame {frame_idx}"
        if total_frames > 0:
            txt += f" / {total_frames} ({int(progress * 100)}%)"
        self.progress_label.configure(text=txt, text_color=TEXT_PRIMARY)

    def _add_violation_item(self, violation, timestamp_sec: float) -> None:
        mins = int(timestamp_sec // 60)
        secs = int(timestamp_sec % 60)
        time_str = f"{mins:02d}:{secs:02d}"

        card = ctk.CTkFrame(
            self.violations_scroll,
            fg_color=BG_CARD,
            corner_radius=6,
            border_width=1,
            border_color="#7f1d1d",
        )
        card.pack(fill="x", pady=2, padx=2)

        icon_lbl = ctk.CTkLabel(card, text="🔴", font=("Segoe UI", 12))
        icon_lbl.pack(side="left", padx=(8, 4), pady=4)

        info_lbl = ctk.CTkLabel(
            card,
            text=f"[{time_str}] {violation.description}",
            font=FONT_BODY_BOLD,
            text_color=COLOR_DANGER,
        )
        info_lbl.pack(side="left", pady=4)

        conf_lbl = ctk.CTkLabel(
            card,
            text=f"Conf: {int(violation.confidence * 100)}%",
            font=FONT_SMALL,
            text_color=TEXT_MUTED,
        )
        conf_lbl.pack(side="right", padx=8, pady=4)

        self.status_badge.configure(text="VIOLAÇÃO 🚨", fg_color=COLOR_DANGER)

    def _on_analysis_finished(self, total_viols: int, elapsed: float, total_frames: int) -> None:
        self.is_processing = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_bar.set(1.0)

        if total_viols > 0:
            self.status_badge.configure(text=f"{total_viols} INFRAÇÕES", fg_color=COLOR_DANGER)
            msg = f"⚠️ Concluído em {elapsed:.1f}s ({total_frames} frames). {total_viols} infração(ões) registrada(s)."
            self.progress_label.configure(text=msg, text_color=COLOR_DANGER)
            self.summary_label.configure(
                text=f"Evidências visuais gravadas na pasta de alertas.",
                text_color=TEXT_SECONDARY,
            )
        else:
            self.status_badge.configure(text="CONFORME ✅", fg_color=COLOR_SUCCESS)
            msg = f"✅ Vídeo analisado em {elapsed:.1f}s com 100% de conformidade de EPI!"
            self.progress_label.configure(text=msg, text_color=COLOR_SUCCESS)
            self.summary_label.configure(
                text="Nenhuma infração identificada durante a gravação.",
                text_color=COLOR_SUCCESS,
            )

    def _on_analysis_failed(self, error_msg: str) -> None:
        self.is_processing = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_badge.configure(text="ERRO ❌", fg_color=COLOR_DANGER)
        self.progress_label.configure(text=f"❌ Falha: {error_msg}", text_color=COLOR_DANGER)
        self.video_canvas.show_placeholder(f"Erro na análise:\n{error_msg}")

    def _on_analysis_stopped(self) -> None:
        self.is_processing = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_badge.configure(text="INTERROMPIDO", fg_color=COLOR_WARNING)
        self.progress_label.configure(text="Análise interrompida pelo usuário.", text_color=COLOR_WARNING)

    def _on_open_evidence_folder(self) -> None:
        """Abre a pasta de evidências no Windows Explorer."""
        folder = Path(self.container.settings.alerts.save_dir).resolve()
        if self._current_metadata and self._current_metadata.plate:
            sub = folder / self._current_metadata.plate / self._current_metadata.record_date
            if sub.exists():
                folder = sub
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)
