"""Motor de conformidade para verificação de uso de EPI.

Analisa as detecções do YOLO e determina se há violações
de segurança (ausência de EPI obrigatório), aplicando
lógica de frames consecutivos para evitar falsos positivos.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """Representa uma violação de segurança confirmada."""
    violation_type: str          # Tipo da violação (ex: 'no_helmet')
    track_id: Optional[int]      # ID do trabalhador rastreado
    timestamp: float             # Timestamp Unix da detecção
    consecutive_frames: int      # Número de frames consecutivos com a violação
    confidence: float            # Confiança média da detecção
    bbox: tuple                  # Última posição (bounding box)

    @property
    def description(self) -> str:
        """Descrição legível da violação em português."""
        names = {
            "no_helmet": "Sem Capacete 🔴",
            "no_gloves": "Sem Luvas 🔴",
            "no_boots": "Sem Botas 🔴",
            "no_goggle": "Sem Óculos de Proteção 🔴",
            "no_vest": "Sem Colete 🔴",
        }
        return names.get(self.violation_type, self.violation_type)


class ComplianceChecker:
    """Verificador de conformidade de uso de EPI.
    
    Monitora detecções ao longo do tempo e confirma violações
    somente quando elas persistem por um número mínimo de frames
    consecutivos, reduzindo falsos positivos.
    """

    def __init__(
        self,
        required_ppe: Optional[List[str]] = None,
        violation_classes: Optional[List[str]] = None,
        min_violation_frames: int = 5,
    ):
        """Inicializa o verificador de conformidade.
        
        Args:
            required_ppe: Lista de EPIs obrigatórios (ex: ['helmet', 'gloves'])
            violation_classes: Classes que indicam ausência de EPI
            min_violation_frames: Frames consecutivos mínimos para confirmar violação
        """
        self.required_ppe = set(required_ppe or ["helmet", "gloves", "goggles", "vest", "boots"])
        self.violation_classes = set(violation_classes or [
            "no_helmet", "no_gloves", "no_boots", "no_goggle"
        ])
        self.min_violation_frames = min_violation_frames

        # Rastreamento de violações: {(track_id, violation_type): frame_count}
        self._violation_counter: Dict[tuple, int] = defaultdict(int)
        # Confiança acumulada para média
        self._confidence_sum: Dict[tuple, float] = defaultdict(float)
        # Última bbox por violação
        self._last_bbox: Dict[tuple, tuple] = {}
        # Conjunto de chaves vistas no frame atual (para resetar contadores)
        self._current_frame_keys: Set[tuple] = set()
        # Violações já confirmadas (para não repetir)
        self._confirmed_violations: Set[tuple] = set()

        logger.info(f"⚖️ Compliance inicializado. EPIs obrigatórios: {self.required_ppe}")
        logger.info(f"⚖️ Frames mínimos para violação: {self.min_violation_frames}")

    def check(self, detections: list) -> List[Violation]:
        """Verifica violações nas detecções do frame atual.
        
        Args:
            detections: Lista de Detection do detector.py
            
        Returns:
            Lista de violações CONFIRMADAS (que atingiram min_violation_frames).
        """
        if not detections:
            return []

        confirmed = []
        self._current_frame_keys.clear()

        for det in detections:
            # Verificar apenas classes de violação relevantes
            if det.class_name not in self.violation_classes:
                continue

            # Verificar se o EPI correspondente é obrigatório
            ppe_type = det.class_name.replace("no_", "")
            # Mapear 'goggle' -> 'goggles' para compatibilidade
            ppe_type_check = ppe_type + "s" if ppe_type == "goggle" else ppe_type
            if ppe_type_check not in self.required_ppe:
                continue

            # Chave única: combinação de track_id e tipo de violação
            key = (det.track_id, det.class_name)
            self._current_frame_keys.add(key)

            # Incrementar contador de frames consecutivos
            self._violation_counter[key] += 1
            self._confidence_sum[key] += det.confidence
            self._last_bbox[key] = det.bbox

            count = self._violation_counter[key]

            # Verificar se atingiu o mínimo de frames
            if count >= self.min_violation_frames and key not in self._confirmed_violations:
                avg_conf = self._confidence_sum[key] / count
                violation = Violation(
                    violation_type=det.class_name,
                    track_id=det.track_id,
                    timestamp=time.time(),
                    consecutive_frames=count,
                    confidence=avg_conf,
                    bbox=det.bbox,
                )
                confirmed.append(violation)
                self._confirmed_violations.add(key)
                logger.warning(
                    f"🚨 VIOLAÇÃO CONFIRMADA: {violation.description} "
                    f"(ID: {det.track_id}, Frames: {count}, Conf: {avg_conf:.2f})"
                )

        # Resetar contadores de violações que NÃO apareceram neste frame
        keys_to_reset = [
            k for k in self._violation_counter
            if k not in self._current_frame_keys
        ]
        for key in keys_to_reset:
            del self._violation_counter[key]
            self._confidence_sum.pop(key, None)
            self._last_bbox.pop(key, None)
            self._confirmed_violations.discard(key)

        return confirmed

    def get_active_violations_count(self) -> int:
        """Retorna o número de violações sendo rastreadas (não confirmadas ainda)."""
        return len(self._violation_counter)

    def reset(self):
        """Reseta todos os contadores e violações."""
        self._violation_counter.clear()
        self._confidence_sum.clear()
        self._last_bbox.clear()
        self._current_frame_keys.clear()
        self._confirmed_violations.clear()
        logger.info("🔄 Compliance resetado.")
