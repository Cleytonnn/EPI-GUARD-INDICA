"""Tela de Revisão para Treino (Aprendizado Ativo).

Exibe os frames coletados automaticamente durante as análises com os
pré-rótulos da IA desenhados. O revisor confirma ou corrige cada amostra
com um clique (ou atalho de teclado) e exporta o dataset para retreino.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

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
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

REASON_LABELS = {
    "violacao": "🚨 Gerou alerta",
    "incerto": "❓ Confiança duvidosa",
    "aleatorio": "🎲 Amostra aleatória",
}

# Cores BGR para desenhar os pré-rótulos
BOX_COLORS = {
    "Person": (229, 136, 30),
    "civil": (160, 160, 160),
}
VIOLATION_COLOR = (68, 68, 239)
PPE_COLOR = (129, 185, 16)


class TrainingReviewView(ctk.CTkFrame):
    """Fila de revisão humana das amostras coletadas para retreino da IA."""

    def __init__(self, master, container: Container, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.container = container
        self.collector = container.training_collector
        self.samples: List[Dict[str, Any]] = []
        self.current: Optional[Dict[str, Any]] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._list_buttons: Dict[str, ctk.CTkButton] = {}

        self._build_ui()
        self._bind_shortcuts()
        self._refresh()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 10))
        ctk.CTkLabel(header, text="🧠 Revisão para Treino da IA", font=FONT_TITLE, text_color=TEXT_PRIMARY).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text=(
                "Frames duvidosos e alertas coletados automaticamente. Confirme ou corrija cada um — "
                "as amostras aprovadas viram dados de treino."
            ),
            font=FONT_BODY,
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", pady=(2, 0))

        toolbar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        toolbar.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkButton(toolbar, text="🔄 Atualizar", font=FONT_BODY, width=110, command=self._refresh).pack(
            side="left", padx=(15, 8), pady=10
        )
        ctk.CTkButton(
            toolbar,
            text="📂 Abrir Pasta",
            font=FONT_BODY,
            width=120,
            fg_color="#334155",
            hover_color="#475569",
            command=self._open_folder,
        ).pack(side="left", padx=(0, 8), pady=10)
        self.export_btn = ctk.CTkButton(
            toolbar,
            text="📦 Exportar Dataset de Treino",
            font=FONT_BODY_BOLD,
            fg_color=COLOR_SUCCESS,
            hover_color="#059669",
            command=self._export_dataset,
        )
        self.export_btn.pack(side="left", padx=(0, 8), pady=10)

        self.stats_lbl = ctk.CTkLabel(toolbar, text="", font=FONT_BODY_BOLD, text_color=TEXT_PRIMARY)
        self.stats_lbl.pack(side="right", padx=15)

        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 15))
        content.grid_columnconfigure(0, weight=2)
        content.grid_columnconfigure(1, weight=5)
        content.grid_rowconfigure(0, weight=1)

        # Lista de pendentes
        list_card = ctk.CTkFrame(content, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        list_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        list_card.grid_columnconfigure(0, weight=1)
        list_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(list_card, text="⏳ Pendentes de Revisão", font=FONT_BODY_BOLD, text_color=TEXT_PRIMARY).grid(
            row=0, column=0, sticky="w", padx=15, pady=(10, 5)
        )
        self.list_scroll = ctk.CTkScrollableFrame(
            list_card, fg_color=BG_INPUT, corner_radius=8, border_width=1, border_color=BORDER_COLOR
        )
        self.list_scroll.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Visualizador + ações
        preview = ctk.CTkFrame(content, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        preview.grid(row=0, column=1, sticky="nsew")
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(2, weight=1)

        self.info_title = ctk.CTkLabel(preview, text="Nenhuma amostra selecionada", font=FONT_BODY_BOLD, text_color=TEXT_PRIMARY)
        self.info_title.grid(row=0, column=0, sticky="w", padx=15, pady=(10, 0))
        self.info_detail = ctk.CTkLabel(preview, text="", font=FONT_SMALL, text_color=TEXT_SECONDARY, justify="left", anchor="w")
        self.info_detail.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 5))

        self.canvas = ctk.CTkCanvas(preview, bg=BG_INPUT, highlightthickness=0)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=15, pady=(0, 10))

        actions = ctk.CTkFrame(preview, fg_color="transparent")
        actions.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 6))
        for col in range(4):
            actions.grid_columnconfigure(col, weight=1)

        self.action_buttons = [
            ctk.CTkButton(
                actions, text="✅ Correto  [C]", font=FONT_BODY_BOLD, fg_color=COLOR_SUCCESS, hover_color="#059669",
                height=36, command=lambda: self._review("correto"),
            ),
            ctk.CTkButton(
                actions, text="🟢 Falso Alarme  [F]", font=FONT_BODY_BOLD, fg_color=COLOR_PRIMARY, hover_color="#1565c0",
                height=36, command=lambda: self._review("falso_alarme"),
            ),
            ctk.CTkButton(
                actions, text="👤 Civil / Pedestre  [V]", font=FONT_BODY_BOLD, fg_color="#475569", hover_color="#64748b",
                height=36, command=lambda: self._review("civil"),
            ),
            ctk.CTkButton(
                actions, text="🗑️ Descartar  [D]", font=FONT_BODY_BOLD, fg_color=COLOR_DANGER, hover_color="#b91c1c",
                height=36, command=lambda: self._review("descartar"),
            ),
        ]
        for col, btn in enumerate(self.action_buttons):
            btn.grid(row=0, column=col, sticky="ew", padx=3)

        ctk.CTkLabel(
            preview,
            text=(
                "Correto = caixas da IA estão certas  •  Falso Alarme = a pessoa ESTAVA com o EPI (inverte 'sem EPI')  •  "
                "Civil = pessoas não são funcionários  •  Descartar = frame ruim/ambíguo"
            ),
            font=FONT_SMALL,
            text_color=TEXT_MUTED,
            wraplength=760,
            justify="left",
        ).grid(row=4, column=0, sticky="w", padx=15, pady=(0, 10))

        self.status_lbl = ctk.CTkLabel(preview, text="", font=FONT_SMALL, text_color=TEXT_SECONDARY, anchor="w")
        self.status_lbl.grid(row=5, column=0, sticky="ew", padx=15, pady=(0, 10))

        self._set_actions_state("disabled")

    def _bind_shortcuts(self) -> None:
        top = self.winfo_toplevel()
        mapping = {"c": "correto", "f": "falso_alarme", "v": "civil", "d": "descartar"}
        for key, action in mapping.items():
            for k in (key, key.upper()):
                top.bind(f"<KeyPress-{k}>", lambda e, a=action: self._on_shortcut(e, a), add="+")

    def _on_shortcut(self, event, action: str) -> None:
        # Só reage com a tela visível e fora de campos de texto
        if not self.winfo_ismapped() or self.current is None:
            return
        if event.widget.winfo_class() in ("Entry", "Text", "TEntry"):
            return
        self._review(action)

    # ------------------------------------------------------------------
    # Dados
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        for child in self.list_scroll.winfo_children():
            child.destroy()
        self._list_buttons = {}

        self.samples = self.collector.list_samples(limit=300)
        self._update_stats()

        if not self.collector.enabled:
            ctk.CTkLabel(
                self.list_scroll,
                text="Coleta desativada.\nAtive 'active_learning.enabled'\nem config/settings.yaml.",
                text_color=TEXT_MUTED,
            ).pack(pady=20)

        if not self.samples:
            if self.collector.enabled:
                ctk.CTkLabel(
                    self.list_scroll,
                    text="Nenhuma amostra pendente.\nAnalise vídeos para coletar\nframes automaticamente.",
                    text_color=TEXT_MUTED,
                ).pack(pady=20)
            self._show_sample(None)
            return

        for sample in self.samples:
            reasons = " ".join(REASON_LABELS.get(r, r).split(" ")[0] for r in sample.get("reasons", []))
            t = sample.get("video_time") or ""
            src = Path(sample.get("source") or "").stem[:22]
            btn = ctk.CTkButton(
                self.list_scroll,
                text=f"{reasons}  {src}  {t}",
                font=FONT_SMALL,
                fg_color=BG_CARD,
                hover_color="#334155",
                anchor="w",
                height=30,
                command=lambda s=sample: self._show_sample(s),
            )
            btn.pack(fill="x", pady=2, padx=2)
            self._list_buttons[sample["id"]] = btn

        self._show_sample(self.samples[0])

    def _update_stats(self) -> None:
        st = self.collector.stats()
        self.stats_lbl.configure(
            text=f"⏳ {st['pendentes']} pendentes   ✅ {st['aprovados']} aprovadas   🗑️ {st['rejeitados']} descartadas"
        )

    def _show_sample(self, sample: Optional[Dict[str, Any]]) -> None:
        self.current = sample
        self.canvas.delete("all")
        if sample is None:
            self.info_title.configure(text="Nenhuma amostra selecionada")
            self.info_detail.configure(text="")
            self._set_actions_state("disabled")
            return

        reasons = ", ".join(REASON_LABELS.get(r, r) for r in sample.get("reasons", []))
        viols = ", ".join(sample.get("violations", [])) or "—"
        self.info_title.configure(text=f"{Path(sample.get('source') or '').name}  •  {sample.get('video_time') or ''}")
        self.info_detail.configure(
            text=(
                f"Placa: {sample.get('plate') or '—'}   |   Motivo: {reasons}   |   Alertas: {viols}   |   "
                f"Caixas: {len(sample.get('detections', []))}"
            )
        )
        self._set_actions_state("normal")
        self._draw_sample(sample)

    def _draw_sample(self, sample: Dict[str, Any]) -> None:
        img = cv2.imread(sample["image_path"])
        if img is None:
            self.canvas.create_text(200, 150, text="Erro ao carregar imagem", fill="#ef4444")
            return

        thickness = max(2, img.shape[1] // 640)
        for d in sample.get("detections", []):
            name = d["class_name"]
            if name in BOX_COLORS:
                color = BOX_COLORS[name]
            elif name.startswith("no_"):
                color = VIOLATION_COLOR
            else:
                color = PPE_COLOR
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            label = f"{name} {int(d['confidence'] * 100)}%"
            cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thickness / 2 + 0.2, color, thickness)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cw = max(self.canvas.winfo_width(), 500)
        ch = max(self.canvas.winfo_height(), 320)
        scale = min(cw / pil.width, ch / pil.height)
        pil = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")

    # ------------------------------------------------------------------
    # Ações
    # ------------------------------------------------------------------
    def _review(self, action: str) -> None:
        if self.current is None:
            return
        sample_id = self.current["id"]
        try:
            dest = self.collector.review(sample_id, action)
        except Exception as e:
            self.status_lbl.configure(text=f"❌ Erro ao revisar: {e}", text_color=COLOR_DANGER)
            return

        idx = next((i for i, s in enumerate(self.samples) if s["id"] == sample_id), 0)
        self.samples = [s for s in self.samples if s["id"] != sample_id]
        btn = self._list_buttons.pop(sample_id, None)
        if btn is not None:
            btn.destroy()

        self.status_lbl.configure(text=f"✔️ {sample_id} → {dest}", text_color=COLOR_SUCCESS)
        self._update_stats()
        if self.samples:
            self._show_sample(self.samples[min(idx, len(self.samples) - 1)])
        else:
            self._refresh()

    def _set_actions_state(self, state: str) -> None:
        for btn in self.action_buttons:
            btn.configure(state=state)

    def _open_folder(self) -> None:
        p = Path(self.collector.output_dir).resolve()
        p.mkdir(parents=True, exist_ok=True)
        os.startfile(p)

    def _export_dataset(self) -> None:
        self.export_btn.configure(state="disabled")
        self.status_lbl.configure(text="📦 Exportando dataset...", text_color=COLOR_WARNING)

        def worker():
            try:
                result = self.collector.export_dataset()
                self.after(0, self._on_export_done, result, None)
            except Exception as e:
                self.after(0, self._on_export_done, None, str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_export_done(self, result: Optional[Dict[str, Any]], error: Optional[str]) -> None:
        self.export_btn.configure(state="normal")
        if error:
            self.status_lbl.configure(text=f"❌ Falha ao exportar: {error}", text_color=COLOR_DANGER)
            return
        if not result or not result.get("data_yaml"):
            self.status_lbl.configure(text="⚠️ Nenhuma amostra aprovada para exportar ainda.", text_color=COLOR_WARNING)
            return
        self.status_lbl.configure(
            text=(
                f"✅ Dataset: {result['train']} treino / {result['val']} validação. Para treinar: "
                f"python scripts/train_model.py --data \"{result['data_yaml']}\""
            ),
            text_color=COLOR_SUCCESS,
        )
