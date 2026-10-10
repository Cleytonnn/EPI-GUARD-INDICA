"""Utilitários de visualização e desenho para o sistema de detecção de EPI.

Funções para desenhar bounding boxes, labels e informações de status
nos frames de vídeo processados.
"""

import cv2
import time
import logging
import numpy as np
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cores (BGR)
COLOR_GREEN = (0, 200, 0)       # EPI detectado ✅
COLOR_RED = (0, 0, 220)         # Violação ❌
COLOR_YELLOW = (0, 220, 220)    # Aviso ⚠️
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (200, 100, 0)      # Pessoa detectada


class FPSCounter:
    """Contador de FPS (Frames Por Segundo) com média móvel."""

    def __init__(self, avg_window: int = 30):
        """Inicializa o contador de FPS.
        
        Args:
            avg_window: Número de frames para calcular a média.
        """
        self._start_time = time.time()
        self._frame_count = 0
        self._fps = 0.0
        self._avg_window = avg_window
        self._times: list = []

    def update(self) -> float:
        """Atualiza o contador e retorna o FPS atual.
        
        Returns:
            FPS atual (média móvel).
        """
        now = time.time()
        self._times.append(now)
        self._frame_count += 1

        # Manter apenas a janela de tempo
        if len(self._times) > self._avg_window:
            self._times = self._times[-self._avg_window:]

        if len(self._times) >= 2:
            elapsed = self._times[-1] - self._times[0]
            if elapsed > 0:
                self._fps = (len(self._times) - 1) / elapsed

        return self._fps

    @property
    def fps(self) -> float:
        """Retorna o FPS atual."""
        return self._fps


def draw_detections(
    frame: np.ndarray,
    detections: list,
    bbox_thickness: int = 2,
) -> np.ndarray:
    """Desenha bounding boxes e labels das detecções no frame.
    
    Args:
        frame: Frame do vídeo (numpy array BGR).
        detections: Lista de objetos Detection.
        bbox_thickness: Espessura das caixas delimitadoras.
        
    Returns:
        Frame com as anotações desenhadas.
    """
    annotated = frame.copy()

    for det in detections:
        # Escolher cor baseada no tipo de detecção
        if det.is_violation:
            color = COLOR_RED
        elif det.class_name == "Person":
            color = COLOR_BLUE
        elif det.class_name in ("helmet", "gloves", "vest", "boots", "goggles"):
            color = COLOR_GREEN
        else:
            color = COLOR_YELLOW

        x1, y1, x2, y2 = det.bbox

        # Desenhar bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, bbox_thickness)

        # Montar label
        label_parts = [_get_label_pt(det.class_name)]
        label_parts.append(f"{det.confidence:.0%}")
        if det.track_id is not None:
            label_parts.append(f"ID:{det.track_id}")
        label = " | ".join(label_parts)

        # Desenhar background do label
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(y1 - 10, label_h + 5)
        cv2.rectangle(
            annotated,
            (x1, label_y - label_h - 5),
            (x1 + label_w + 5, label_y + baseline),
            color,
            -1,  # Preenchido
        )

        # Desenhar texto do label
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_status_bar(
    frame: np.ndarray,
    fps: float = 0.0,
    violations_count: int = 0,
    alerts_sent: int = 0,
    show_fps: bool = True,
) -> np.ndarray:
    """Desenha uma barra de status no topo do frame.
    
    Args:
        frame: Frame do vídeo.
        fps: FPS atual.
        violations_count: Número de violações ativas.
        alerts_sent: Total de alertas enviados.
        show_fps: Se True, mostra o FPS.
        
    Returns:
        Frame com barra de status.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    bar_height = 35

    # Background da barra (semi-transparente)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

    # Textos da barra
    x_offset = 10
    y_pos = 25

    # Status do sistema
    status_color = COLOR_RED if violations_count > 0 else COLOR_GREEN
    status_text = "● VIOLAÇÃO DETECTADA" if violations_count > 0 else "● MONITORANDO"
    cv2.putText(
        annotated, status_text, (x_offset, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA,
    )
    x_offset += 280

    # FPS
    if show_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            annotated, fps_text, (x_offset, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA,
        )
        x_offset += 120

    # Alertas
    alert_text = f"Alertas: {alerts_sent}"
    cv2.putText(
        annotated, alert_text, (x_offset, y_pos),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA,
    )

    return annotated


def draw_violation_warning(
    frame: np.ndarray,
    violation_text: str,
) -> np.ndarray:
    """Desenha um aviso grande de violação no frame.
    
    Args:
        frame: Frame do vídeo.
        violation_text: Texto da violação.
        
    Returns:
        Frame com aviso de violação.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Background do aviso (semi-transparente)
    warning_h = 50
    y_start = h - warning_h - 10
    overlay = annotated.copy()
    cv2.rectangle(overlay, (10, y_start), (w - 10, y_start + warning_h), COLOR_RED, -1)
    cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

    # Borda
    cv2.rectangle(annotated, (10, y_start), (w - 10, y_start + warning_h), COLOR_RED, 2)

    # Texto centralizado
    text = f"⚠ {violation_text}"
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    text_x = (w - text_w) // 2
    text_y = y_start + (warning_h + text_h) // 2
    cv2.putText(
        annotated, text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2, cv2.LINE_AA,
    )

    return annotated


def draw_vehicle_metadata(
    frame: np.ndarray,
    placa: str,
    data: str,
    video_name: str,
    video_time: str,
) -> np.ndarray:
    """Desenha tarja superior com dados do veículo e da gravação no frame.
    
    Args:
        frame: Frame do vídeo.
        placa: Placa do veículo identificada pela pasta.
        data: Data da gravação identificada pela pasta.
        video_name: Nome do arquivo MP4.
        video_time: Timestamp no vídeo (ex: 00:01:45).
        
    Returns:
        Frame anotado com os metadados do veículo.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    badge_height = 30

    # Fundo semitransparente no topo ou abaixo da barra de status
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, badge_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Texto com as informações
    info_text = f"VEICULO: {placa.upper()}  |  DATA: {data}  |  VIDEO: {video_name}  |  TEMPO: {video_time}"
    cv2.putText(
        annotated,
        info_text,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),  # Amarelo
        1,
        cv2.LINE_AA,
    )

    return annotated


def _get_label_pt(class_name: str) -> str:
    """Converte nome da classe para português.
    
    Args:
        class_name: Nome da classe em inglês.
        
    Returns:
        Nome traduzido para português.
    """
    translations = {
        "helmet": "Capacete ✅",
        "gloves": "Luvas ✅",
        "vest": "Colete ✅",
        "boots": "Botas ✅",
        "goggles": "Óculos ✅",
        "no_helmet": "SEM CAPACETE ❌",
        "no_gloves": "SEM LUVAS ❌",
        "no_boots": "SEM BOTAS ❌",
        "no_goggle": "SEM ÓCULOS ❌",
        "no_vest": "SEM COLETE ❌",
        "Person": "Pessoa",
        "none": "---",
    }
    return translations.get(class_name, class_name)


def setup_logging(level: int = logging.INFO) -> None:
    """Configura o logging do sistema.
    
    Args:
        level: Nível de logging (ex: logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Reduzir verbosidade de bibliotecas externas
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

