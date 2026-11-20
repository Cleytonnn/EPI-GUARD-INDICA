"""Componentes visuais e renderizadores de tela para a camada de apresentação."""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.core.domain.detection import Detection

COLOR_GREEN = (0, 200, 0)
COLOR_RED = (0, 0, 220)
COLOR_YELLOW = (0, 220, 220)
COLOR_WHITE = (255, 255, 255)
COLOR_BLUE = (200, 100, 0)


class FPSCounter:
    """Calculador de FPS com média móvel."""

    def __init__(self, avg_window: int = 30):
        self._avg_window = avg_window
        self._times: list[float] = []
        self._fps: float = 0.0

    def update(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._avg_window:
            self._times = self._times[-self._avg_window :]

        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            if elapsed > 0:
                self._fps = (len(self._times) - 1) / elapsed

        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def _ascii_label(text: str) -> str:
    """Remove acentos e emojis incompatíveis com cv2.putText (evita '?' nas caixas)."""
    import unicodedata

    norm = unicodedata.normalize("NFKD", text)
    ascii_only = norm.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.split())


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    bbox_thickness: int = 2,
) -> np.ndarray:
    """Desenha caixas delimitadoras e rótulos das detecções no frame."""
    annotated = frame.copy()
    placed_labels: list[tuple[int, int, int, int]] = []

    for det in detections:
        if det.class_name == "civil":
            color = (130, 130, 130)  # Cinza neutro para pedestres/civis na rua
        elif det.is_violation:
            color = COLOR_RED
        elif det.class_name == "Person":
            color = COLOR_BLUE
        elif det.class_name in ("helmet", "gloves", "vest", "boots", "goggles", "uniform_rf"):
            color = COLOR_GREEN
        else:
            color = COLOR_YELLOW

        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, bbox_thickness)

        label_parts = [_ascii_label(det.label_pt)]
        if det.track_id is not None:
            label_parts.append(f"#{det.track_id}")
        label_parts.append(f"{det.confidence:.0%}")
        label = " | ".join(label_parts)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        box_h = txt_h + baseline + 4
        label_y2 = max(box_h, y1)
        label_y1 = label_y2 - box_h
        lx2 = x1 + txt_w + 6

        # Evita sobreposição vertical quando dois rótulos coincidem na mesma posição
        for _ in range(5):
            overlap = any(
                not (lx2 < px1 or x1 > px2 or label_y2 < py1 or label_y1 > py2) for px1, py1, px2, py2 in placed_labels
            )
            if not overlap:
                break
            if label_y1 - box_h - 2 >= 0:
                label_y1 -= box_h + 2
                label_y2 -= box_h + 2
            else:
                label_y1 += box_h + 2
                label_y2 += box_h + 2

        placed_labels.append((x1, label_y1, lx2, label_y2))
        cv2.rectangle(annotated, (x1, label_y1), (lx2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            font,
            font_scale,
            COLOR_WHITE,
            font_thick,
            cv2.LINE_AA,
        )

    return annotated


def draw_vehicle_metadata(
    frame: np.ndarray,
    plate: str,
    record_date: str,
    video_name: str,
    video_time: str,
) -> np.ndarray:
    """Desenha tarja superior com informações do veículo e momento do vídeo."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    badge_height = 30

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, badge_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    text = f"VEICULO: {plate.upper()}  |  DATA: {record_date}  |  ARQUIVO: {video_name}  |  TEMPO: {video_time}"
    cv2.putText(
        annotated,
        text,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated


def draw_status_bar(
    frame: np.ndarray,
    fps: float,
    violations_count: int,
    alerts_sent: int,
) -> np.ndarray:
    """Desenha barra de status inferior informativa."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    bar_height = 35

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    fps_text = f"FPS: {fps:.1f}"
    viol_color = COLOR_GREEN if violations_count == 0 else COLOR_RED
    viol_text = f"Violacoes: {violations_count}"
    alerts_text = f"Alertas: {alerts_sent}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = h - 11

    cv2.putText(annotated, fps_text, (10, y_pos), font, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(annotated, viol_text, (150, y_pos), font, 0.55, viol_color, 1, cv2.LINE_AA)
    cv2.putText(annotated, alerts_text, (330, y_pos), font, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    sig = "CRIADO POR CLEYTON BAUMGRATZ"
    (sig_w, _), _ = cv2.getTextSize(sig, font, 0.45, 1)
    if w - sig_w - 12 > 460:
        cv2.putText(annotated, sig, (w - sig_w - 12, y_pos), font, 0.45, (180, 220, 255), 1, cv2.LINE_AA)

    return annotated


def draw_violation_warning(frame: np.ndarray, violation_text: str) -> np.ndarray:
    """Desenha caixa de alerta vermelha em destaque na parte inferior do frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    warning_h = 50
    y_start = h - warning_h - 40
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, y_start), (w - 10, y_start + warning_h), COLOR_RED, -1)
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    cv2.rectangle(annotated, (10, y_start), (w - 10, y_start + warning_h), COLOR_RED, 2)

    text = f"⚠ {violation_text}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = (w - text_w) // 2
    text_y = y_start + (warning_h + text_h) // 2
    cv2.putText(
        annotated,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_WHITE,
        2,
        cv2.LINE_AA,
    )
    return annotated
