"""Widget para renderização fluida de frames de vídeo com redimensionamento proporcional."""

from __future__ import annotations

import tkinter as tk
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.presentation.desktop.theme import BG_CARD, BORDER_COLOR, TEXT_MUTED


class VideoCanvas(tk.Canvas):
    """Canvas de alto desempenho para exibição de vídeo com aspect-ratio mantido."""

    def __init__(self, master, width: int = 640, height: int = 360, **kwargs):
        super().__init__(
            master,
            width=width,
            height=height,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            **kwargs,
        )
        self.target_width = width
        self.target_height = height
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._image_item: Optional[int] = None
        self._placeholder_text: Optional[int] = None
        self.show_placeholder("Nenhum vídeo em reprodução")

        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event) -> None:
        if event.width > 50 and event.height > 50:
            self.target_width = event.width
            self.target_height = event.height
            if self._placeholder_text is not None:
                self.coords(self._placeholder_text, self.target_width // 2, self.target_height // 2)

    def show_placeholder(self, text: str = "Aguardando vídeo...") -> None:
        """Exibe uma mensagem centralizada quando não há vídeo."""
        self.delete("all")
        self._photo_image = None
        self._image_item = None
        self._placeholder_text = self.create_text(
            self.target_width // 2,
            self.target_height // 2,
            text=f"🎥\n\n{text}",
            fill=TEXT_MUTED,
            font=("Segoe UI", 12),
            justify="center",
        )

    def update_frame(self, frame: np.ndarray) -> None:
        """Recebe um frame OpenCV (BGR) e atualiza o Canvas de forma proporcional."""
        if frame is None or frame.size == 0:
            return

        try:
            h, w = frame.shape[:2]
            # Calcular dimensões mantendo proporção original
            canvas_w = max(self.target_width, 100)
            canvas_h = max(self.target_height, 100)

            scale = min(canvas_w / w, canvas_h / h)
            new_w = max(int(w * scale), 1)
            new_h = max(int(h * scale), 1)

            # Redimensionar com OpenCV para alta velocidade
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(rgb)
            self._photo_image = ImageTk.PhotoImage(pil_img)

            if self._placeholder_text is not None:
                self.delete(self._placeholder_text)
                self._placeholder_text = None

            pos_x = canvas_w // 2
            pos_y = canvas_h // 2

            if self._image_item is None:
                self._image_item = self.create_image(pos_x, pos_y, image=self._photo_image, anchor="center")
            else:
                self.coords(self._image_item, pos_x, pos_y)
                self.itemconfig(self._image_item, image=self._photo_image)
        except Exception:
            pass
