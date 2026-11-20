"""Repositório de histórico e relatórios de infrações em formato CSV."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

from src.core.domain.violation import Violation
from src.core.interfaces.report_repo import IViolationReportRepository

logger = logging.getLogger(__name__)


class CsvViolationReportRepository(IViolationReportRepository):
    """Implementa IViolationReportRepository gravando em arquivo CSV com delimitador ';'."""

    def __init__(self, csv_path: str | Path = "relatorio_infracoes.csv"):
        self.csv_path = Path(csv_path)
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Cria o arquivo com cabeçalho se ainda não existir."""
        if not self.csv_path.exists():
            try:
                self.csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f, delimiter=";")
                    writer.writerow(
                        [
                            "Data_Processamento",
                            "Placa_Veiculo",
                            "Data_Gravacao",
                            "Arquivo_Video",
                            "Tempo_Video_Formatado",
                            "Tempo_Video_Segundos",
                            "Tipo_Infracao",
                            "Descricao_Infracao",
                            "Confianca_Percentual",
                            "ID_Trabalhador",
                            "Frames_Consecutivos",
                            "Caminho_Imagem_Alerta",
                        ]
                    )
            except Exception as e:
                logger.error(f"❌ Erro ao criar cabeçalho do CSV: {e}")

    def log_violation(self, violation: Violation, image_path: str | None = None) -> None:
        """Registra uma linha no arquivo CSV de infrações."""
        try:
            meta = violation.metadata
            plate = meta.plate if meta else "N/A"
            rec_date = meta.record_date if meta else "N/A"
            video_name = meta.video_name if meta else "N/A"
            video_time_str = meta.video_time_str if meta else "00:00:00"
            video_second = f"{meta.video_second:.1f}" if meta else "0.0"

            with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        plate,
                        rec_date,
                        video_name,
                        video_time_str,
                        video_second,
                        violation.violation_type,
                        violation.description,
                        f"{violation.confidence:.1%}",
                        violation.track_id if violation.track_id is not None else "N/A",
                        violation.consecutive_frames,
                        image_path or "",
                    ]
                )
        except Exception as e:
            logger.error(f"❌ Erro ao salvar linha no CSV de infrações: {e}")
