"""Módulo de detecção de EPI usando YOLO.

Encapsula o modelo YOLO (Ultralytics) para detecção de equipamentos
de proteção individual em frames de vídeo.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Representa uma detecção individual no frame."""
    class_name: str          # Nome da classe detectada (ex: 'helmet', 'no_helmet')
    confidence: float        # Confiança da detecção (0.0 a 1.0)
    bbox: tuple              # Bounding box (x1, y1, x2, y2)
    track_id: Optional[int] = None  # ID de tracking (se habilitado)

    @property
    def is_violation(self) -> bool:
        """Verifica se esta detecção é uma violação (ausência de EPI)."""
        return self.class_name.startswith("no_")

    @property
    def ppe_type(self) -> str:
        """Retorna o tipo de EPI relacionado."""
        if self.class_name.startswith("no_"):
            return self.class_name[3:]  # 'no_helmet' -> 'helmet'
        return self.class_name


class PPEDetector:
    """Detector de EPI usando modelo YOLO.
    
    Encapsula o carregamento do modelo, inferência e tracking
    de objetos para detecção de equipamentos de proteção.
    """

    # Mapeamento de classes de violação para nomes legíveis em português
    VIOLATION_NAMES: Dict[str, str] = {
        "no_helmet": "Sem Capacete",
        "no_gloves": "Sem Luvas",
        "no_boots": "Sem Botas",
        "no_goggle": "Sem Óculos",
        "no_vest": "Sem Colete",
    }

    # Classes que representam EPI sendo usado corretamente
    PPE_CLASSES = {"helmet", "gloves", "vest", "boots", "goggles"}
    
    # Classes que representam ausência de EPI
    VIOLATION_CLASSES = {"no_helmet", "no_gloves", "no_boots", "no_goggle"}

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        track: bool = True,
    ):
        """Inicializa o detector de EPI.
        
        Args:
            model_path: Caminho para o modelo YOLO (.pt)
            confidence: Threshold de confiança mínima
            iou_threshold: Threshold de IoU para Non-Maximum Suppression
            imgsz: Tamanho da imagem para inferência
            track: Se True, habilita tracking de objetos (ByteTrack)
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.track = track
        self._frame_count = 0

        logger.info(f"🤖 Carregando modelo: {model_path}")
        try:
            self.model = YOLO(model_path)
            # Obter nomes das classes do modelo
            self.class_names = self.model.names
            logger.info(f"🤖 Modelo carregado. Classes: {list(self.class_names.values())}")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            raise

    def detect(self, frame: np.ndarray, skip_frames: int = 1) -> List[Detection]:
        """Executa detecção de EPI em um frame.
        
        Args:
            frame: Frame do vídeo (numpy array BGR)
            skip_frames: Processar a cada N frames (1 = todos)
            
        Returns:
            Lista de detecções encontradas no frame.
        """
        self._frame_count += 1
        
        # Skip frames para performance
        if self._frame_count % skip_frames != 0:
            return []

        try:
            # Executar inferência com ou sem tracking
            if self.track:
                results = self.model.track(
                    frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
            else:
                results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    verbose=False,
                )

            return self._parse_results(results)

        except Exception as e:
            logger.error(f"❌ Erro na detecção: {e}")
            return []

    def _parse_results(self, results) -> List[Detection]:
        """Converte resultados do YOLO em lista de Detection."""
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                # Extrair bounding box
                xyxy = boxes.xyxy[i].cpu().numpy()
                bbox = tuple(xyxy.astype(int))

                # Extrair classe e confiança
                cls_id = int(boxes.cls[i].cpu().numpy())
                conf = float(boxes.conf[i].cpu().numpy())
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # Extrair track_id se disponível
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())

                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                    track_id=track_id,
                ))

        return detections

    def get_violation_name(self, class_name: str) -> str:
        """Retorna o nome legível em português da violação."""
        return self.VIOLATION_NAMES.get(class_name, class_name)

    def reset_tracking(self):
        """Reseta o tracking de objetos."""
        self._frame_count = 0
        if hasattr(self.model, 'predictor') and self.model.predictor is not None:
            if hasattr(self.model.predictor, 'trackers'):
                self.model.predictor.trackers = []
        logger.info("🔄 Tracking resetado.")
