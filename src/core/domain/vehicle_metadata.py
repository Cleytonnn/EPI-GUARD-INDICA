"""Value Object para metadados de veículos e gravações."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class VehicleMetadata:
    """Metadados contextuais extraídos da hierarquia de pastas ou arquivos."""

    plate: str
    record_date: str
    video_name: str
    video_time_str: str = "00:00:00"
    video_second: float = 0.0
    frame_index: int = 0

    @classmethod
    def from_path(cls, video_path: Path | str, root_dir: Path | str | None = None) -> VehicleMetadata:
        """Cria VehicleMetadata a partir da estrutura de diretórios e nome de arquivo.

        Estrutura esperada:
            [root_dir] / [PLACA] / [DATA] / [arquivo.265 / .mp4]
        """
        path_obj = Path(video_path).resolve()
        video_name = path_obj.name

        plate = "DESCONHECIDO"
        record_date = datetime.now().strftime("%Y-%m-%d")

        if root_dir is not None:
            root_obj = Path(root_dir).resolve()
            try:
                rel_parts = path_obj.relative_to(root_obj).parts
                if len(rel_parts) >= 3:
                    plate = rel_parts[0]
                    record_date = rel_parts[1]
                elif len(rel_parts) == 2:
                    plate = rel_parts[0]
                    record_date = datetime.fromtimestamp(path_obj.stat().st_mtime).strftime("%Y-%m-%d")
            except (ValueError, OSError):
                pass

        if plate == "DESCONHECIDO":
            parent_dirs = path_obj.parts
            if len(parent_dirs) >= 3:
                record_date = parent_dirs[-2]
                plate = parent_dirs[-3]
            elif len(parent_dirs) >= 2:
                plate = parent_dirs[-2]
                try:
                    record_date = datetime.fromtimestamp(path_obj.stat().st_mtime).strftime("%Y-%m-%d")
                except OSError:
                    pass

        # Refinamento por padrão de nome de DVR (ex: U20260803-23492916N300.265)
        date_match = re.search(r"[A-Za-z]?(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})", video_name)
        if date_match:
            y, m, d = date_match.group(1), date_match.group(2), date_match.group(3)
            if not re.match(r"^\d{4}[-_]\d{2}[-_]\d{2}$", str(record_date)):
                record_date = f"{y}-{m}-{d}"

        return cls(
            plate=plate.upper(),
            record_date=record_date,
            video_name=video_name,
        )

    def with_time(self, seconds: float, frame_idx: int) -> VehicleMetadata:
        """Retorna uma nova instância com os dados temporais do frame."""
        td = timedelta(seconds=int(seconds))
        tot_secs = int(td.total_seconds())
        h = tot_secs // 3600
        m = (tot_secs % 3600) // 60
        s = tot_secs % 60
        time_str = f"{h:02d}:{m:02d}:{s:02d}"

        return VehicleMetadata(
            plate=self.plate,
            record_date=self.record_date,
            video_name=self.video_name,
            video_time_str=time_str,
            video_second=seconds,
            frame_index=frame_idx,
        )

    @property
    def is_cabin(self) -> bool:
        """Determina se a gravação é proveniente de câmera interna de cabine (CH2, DMS, etc.)."""
        name_upper = self.video_name.upper()
        return any(tag in name_upper for tag in ("CH2", "CABINE", "DMS", "INTERN", "MOTORISTA"))
