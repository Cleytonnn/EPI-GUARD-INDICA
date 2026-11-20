"""Tela de Monitoramento de Câmera ao Vivo (Webcam ou RTSP / Câmera IP)."""

from __future__ import annotations

import threading
import time

import customtkinter as ctk
import cv2

from src.container import Container
from src.presentation.cli.visualizer import draw_detections
from src.presentation.desktop.theme import (
    BG_CARD,
    BG_INPUT,
    BORDER_COLOR,
    COLOR_ACCENT,
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
from src.presentation.desktop.widgets.video_canvas import VideoCanvas


class LiveView(ctk.CTkFrame):
    """View responsável por exibir streams de câmeras em tempo real."""

    def __init__(self, master, container: Container, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.container = container

        self.is_streaming: bool = False
        self.stop_requested: bool = False
        self._cap: cv2.VideoCapture | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # 1. Cabeçalho
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 10))

        ctk.CTkLabel(
            header,
            text="📹 Monitoramento de Câmera ao Vivo",
            font=FONT_TITLE,
            text_color=TEXT_PRIMARY,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Conecte uma webcam local ou stream RTSP de câmera IP para inferência de EPI em tempo real.",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(2, 0))

        # 2. Configurações da Câmera
        config_card = ctk.CTkFrame(
            self,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        config_card.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))
        config_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(config_card, text="Fonte da Câmera:", font=FONT_BODY_BOLD, text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=(15, 10), pady=12
        )

        self.source_entry = ctk.CTkEntry(
            config_card,
            font=FONT_BODY,
            height=32,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
        )
        self.source_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=12)
        self.source_entry.insert(0, "0")

        self.start_btn = ctk.CTkButton(
            config_card,
            text="▶️ Iniciar Câmera",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_PRIMARY,
            hover_color="#1565c0",
            width=120,
            height=32,
            command=self._on_start_stream,
        )
        self.start_btn.grid(row=0, column=2, padx=(0, 5), pady=12)

        self.stop_btn = ctk.CTkButton(
            config_card,
            text="⏹️ Parar",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_DANGER,
            hover_color="#b91c1c",
            width=90,
            height=32,
            state="disabled",
            command=self._on_stop_stream,
        )
        self.stop_btn.grid(row=0, column=3, padx=(5, 15), pady=12)

        # 3. Canvas e Informações
        main_content = ctk.CTkFrame(self, fg_color="transparent")
        main_content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 15))
        main_content.grid_columnconfigure(0, weight=3)
        main_content.grid_columnconfigure(1, weight=1)
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

        p_header = ctk.CTkFrame(player_container, fg_color="transparent")
        p_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        self.stream_status = ctk.CTkLabel(
            p_header,
            text="OFFLINE",
            font=FONT_SMALL,
            fg_color="#334155",
            corner_radius=6,
            padx=8,
            pady=2,
        )
        self.stream_status.pack(side="left")

        self.fps_label = ctk.CTkLabel(p_header, text="0.0 FPS", font=FONT_SMALL, text_color=COLOR_ACCENT)
        self.fps_label.pack(side="right")

        self.video_canvas = VideoCanvas(player_container, width=640, height=400)
        self.video_canvas.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Painel lateral de métricas
        side_panel = ctk.CTkFrame(
            main_content,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_COLOR,
        )
        side_panel.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(side_panel, text="📊 Métricas da Câmera", font=FONT_BODY_BOLD, text_color=TEXT_PRIMARY).pack(
            anchor="w", padx=15, pady=(15, 10)
        )

        self.detections_lbl = ctk.CTkLabel(
            side_panel,
            text="Objetos Detectados: 0",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        )
        self.detections_lbl.pack(anchor="w", padx=15, pady=4)

        self.violations_lbl = ctk.CTkLabel(
            side_panel,
            text="Infrações Ativas: 0",
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        )
        self.violations_lbl.pack(anchor="w", padx=15, pady=4)

        self.device_lbl = ctk.CTkLabel(
            side_panel,
            text=f"Acelerador: {self.container.detector.device.upper()}",
            font=FONT_SMALL,
            text_color=COLOR_SUCCESS,
        )
        self.device_lbl.pack(anchor="w", padx=15, pady=(15, 4))

    def _on_start_stream(self) -> None:
        source_str = self.source_entry.get().strip()
        source = int(source_str) if source_str.isdigit() else source_str

        self.is_streaming = True
        self.stop_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.stream_status.configure(text="CONECTANDO...", fg_color=COLOR_WARNING)

        threading.Thread(target=self._stream_thread, args=(source,), daemon=True).start()

    def _on_stop_stream(self) -> None:
        self.stop_requested = True
        self.stream_status.configure(text="PARANDO...", fg_color=COLOR_WARNING)

    def _stream_thread(self, source) -> None:
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.after(0, self._on_stream_error, f"Não foi possível conectar à câmera: {source}")
                return

            self._cap = cap
            self.after(0, self._set_online_status)

            fps_timer = time.time()
            fps_count = 0
            cur_fps = 0.0
            collector = getattr(self.container, "training_collector", None)
            if collector:
                collector.start_session(f"ao_vivo_{int(fps_timer)}")

            while not self.stop_requested:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                fps_count += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    cur_fps = fps_count / (now - fps_timer)
                    fps_count = 0
                    fps_timer = now

                # Inferência YOLO
                detections = self.container.detector.detect(frame, skip_frames=1)

                # Aprendizado Ativo: guarda frames duvidosos para revisão
                if collector:
                    collector.consider(frame, detections)

                annotated = draw_detections(frame, detections)

                viols_count = sum(1 for d in detections if d.is_violation)
                self.after(0, self._update_stream_ui, annotated, cur_fps, len(detections), viols_count)

            cap.release()
            self._cap = None
            self.after(0, self._on_stream_stopped)

        except Exception as e:
            self.after(0, self._on_stream_error, str(e))

    def _set_online_status(self) -> None:
        self.stream_status.configure(text="ONLINE 🟢", fg_color=COLOR_SUCCESS)

    def _update_stream_ui(self, frame, fps: float, total_dets: int, viols: int) -> None:
        self.video_canvas.update_frame(frame)
        self.fps_label.configure(text=f"{fps:.1f} FPS")
        self.detections_lbl.configure(text=f"Objetos Detectados: {total_dets}")
        if viols > 0:
            self.violations_lbl.configure(text=f"Infrações Ativas: {viols} 🔴", text_color=COLOR_DANGER)
        else:
            self.violations_lbl.configure(text="Infrações Ativas: 0 (OK)", text_color=COLOR_SUCCESS)

    def _on_stream_error(self, err_msg: str) -> None:
        self.is_streaming = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.stream_status.configure(text="ERRO ❌", fg_color=COLOR_DANGER)
        self.video_canvas.show_placeholder(f"Erro na câmera:\n{err_msg}")

    def _on_stream_stopped(self) -> None:
        self.is_streaming = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.stream_status.configure(text="OFFLINE", fg_color="#334155")
        self.video_canvas.show_placeholder("Câmera desligada")
