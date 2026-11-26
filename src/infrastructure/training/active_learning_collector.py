"""Coletor de Aprendizado Ativo (Active Learning).

Durante a análise de vídeos/câmeras, seleciona automaticamente os frames mais
úteis para retreinar a IA, salva o frame ORIGINAL (sem anotações) junto com os
pré-rótulos no formato YOLO e deixa tudo pendente para revisão humana.

Estrutura gerada em ``output_dir`` (padrão ``datasets/coleta_ativa``)::

    pendentes/   <id>.jpg  <id>.txt (pré-rótulo YOLO)  <id>.json (metadados)
    aprovados/   amostras confirmadas/corrigidas pelo revisor
    rejeitados/  amostras descartadas (não entram no treino)
    classes.txt  ordem das classes usada nos arquivos .txt

Critérios de seleção de um frame:
    * ``violacao``  – o frame gerou alerta confirmado (confirma acerto ou revela falso positivo);
    * ``incerto``   – alguma detecção com confiança na faixa duvidosa;
    * ``aleatorio`` – pequena amostra aleatória para não viciar o dataset.

Frames quase idênticos são descartados por hash perceptual (dHash) e há
intervalo mínimo entre coletas do mesmo vídeo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation
from src.core.interfaces.training_collector import ITrainingSampleCollector

logger = logging.getLogger(__name__)

# Classes do dataset de treino (a ordem define o índice no arquivo YOLO .txt).
# "vest"/"no_vest" ficam de fora pois o detector atual as emite duplicadas de uniform_rf.
TRAINING_CLASSES: list[str] = [
    "Person",
    "civil",
    "helmet",
    "no_helmet",
    "uniform_rf",
    "no_uniform_rf",
    "gloves",
    "no_gloves",
]
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(TRAINING_CLASSES)}

# Inversão usada quando o revisor marca "Falso alarme" (a pessoa ESTAVA com o EPI)
FLIP_VIOLATION: dict[str, str] = {
    "no_helmet": "helmet",
    "no_uniform_rf": "uniform_rf",
    "no_gloves": "gloves",
}

STATUS_PENDING = "pendentes"
STATUS_APPROVED = "aprovados"
STATUS_REJECTED = "rejeitados"
ALL_STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED)

REVIEW_ACTIONS = ("correto", "falso_alarme", "civil", "descartar")
_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class _SessionState:
    """Estado de coleta por vídeo/câmera."""

    name: str = "sessao"
    saved: int = 0
    considered: int = 0
    last_saved_frame: int | None = None
    recent_hashes: deque[int] = field(default_factory=lambda: deque(maxlen=60))


class ActiveLearningCollector(ITrainingSampleCollector):
    """Seleciona, salva e gerencia a revisão de amostras para retreino contínuo."""

    def __init__(
        self,
        output_dir: str | Path = "datasets/coleta_ativa",
        enabled: bool = True,
        uncertain_min: float = 0.30,
        uncertain_max: float = 0.60,
        random_rate: float = 0.02,
        min_frame_gap: int = 30,
        max_per_video: int = 150,
        max_pending: int = 5000,
        dedup_threshold: int = 6,
        image_quality: int = 95,
        seed: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.uncertain_min = uncertain_min
        self.uncertain_max = uncertain_max
        self.random_rate = max(0.0, min(1.0, random_rate))
        self.min_frame_gap = max(0, int(min_frame_gap))
        self.max_per_video = max(1, int(max_per_video))
        self.max_pending = max(1, int(max_pending))
        self.dedup_threshold = max(0, int(dedup_threshold))
        self.image_quality = max(1, min(100, int(image_quality)))

        self._rng = random.Random(seed)
        self._lock = threading.Lock()
        self._session = _SessionState()
        self._pending_count: int | None = None

        if self.enabled:
            self._ensure_dirs()

    # ------------------------------------------------------------------
    # Coleta
    # ------------------------------------------------------------------
    def start_session(self, source_name: str) -> None:
        with self._lock:
            self._session = _SessionState(name=source_name or "sessao")

    def consider(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        violations: list[Violation] | None = None,
        metadata: VehicleMetadata | None = None,
    ) -> str | None:
        if not self.enabled or frame is None or frame.size == 0:
            return None

        try:
            with self._lock:
                session = self._session
                session.considered += 1

                if session.saved >= self.max_per_video:
                    return None
                if self._get_pending_count() >= self.max_pending:
                    return None

                reasons = self._selection_reasons(detections, violations)
                if not reasons:
                    return None

                frame_idx = metadata.frame_index if (metadata and metadata.frame_index) else session.considered
                has_violation = "violacao" in reasons
                if (
                    not has_violation
                    and session.last_saved_frame is not None
                    and (frame_idx - session.last_saved_frame) < self.min_frame_gap
                ):
                    return None

                img_hash = self._dhash(frame)
                if any(self._hamming(img_hash, h) <= self.dedup_threshold for h in session.recent_hashes):
                    return None

                sample_id = self._save_sample(frame, detections, violations, metadata, reasons, frame_idx)
                if sample_id is None:
                    return None

                session.saved += 1
                session.last_saved_frame = frame_idx
                session.recent_hashes.append(img_hash)
                if self._pending_count is not None:
                    self._pending_count += 1
                return sample_id
        except Exception as e:  # A coleta nunca pode derrubar a análise
            logger.warning(f"⚠️ Falha na coleta de amostra de treino: {e}")
            return None

    def _selection_reasons(
        self,
        detections: list[Detection],
        violations: list[Violation] | None,
    ) -> list[str]:
        reasons: list[str] = []
        relevant = [d for d in detections if d.class_name in CLASS_INDEX]
        if violations:
            reasons.append("violacao")
        if any(self.uncertain_min <= d.confidence <= self.uncertain_max for d in relevant):
            reasons.append("incerto")
        if relevant and self.random_rate > 0 and self._rng.random() < self.random_rate:
            reasons.append("aleatorio")
        return reasons

    def _save_sample(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        violations: list[Violation] | None,
        metadata: VehicleMetadata | None,
        reasons: list[str],
        frame_idx: int,
    ) -> str | None:
        source = (metadata.video_name if metadata else None) or self._session.name
        stem = self._slug(Path(source).stem)[:40] or "amostra"
        sample_id = f"{stem}_{int(frame_idx):07d}_{uuid4().hex[:6]}"
        folder = self.output_dir / STATUS_PENDING
        h, w = frame.shape[:2]

        ok = cv2.imwrite(
            str(folder / f"{sample_id}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.image_quality],
        )
        if not ok:
            return None

        det_list = [
            {
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 3),
                "bbox": [int(v) for v in d.bbox],
                "track_id": d.track_id,
            }
            for d in detections
            if d.class_name in CLASS_INDEX
        ]

        meta: dict[str, Any] = {
            "id": sample_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "plate": metadata.plate if metadata else None,
            "record_date": metadata.record_date if metadata else None,
            "video_time": metadata.video_time_str if metadata else None,
            "frame_index": int(frame_idx),
            "width": int(w),
            "height": int(h),
            "reasons": reasons,
            "violations": [v.violation_type for v in (violations or [])],
            "detections": det_list,
            "review": None,
        }
        (folder / f"{sample_id}.txt").write_text(self._to_yolo(det_list, w, h), encoding="utf-8")
        (folder / f"{sample_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug(f"🧠 Amostra de treino coletada: {sample_id} ({', '.join(reasons)})")
        return sample_id

    # ------------------------------------------------------------------
    # Revisão humana
    # ------------------------------------------------------------------
    def list_samples(self, status: str = STATUS_PENDING, limit: int | None = None) -> list[dict[str, Any]]:
        """Lista amostras (metadados) de um status, mais recentes primeiro."""
        folder = self.output_dir / status
        if not folder.exists():
            return []
        files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if limit:
            files = files[:limit]
        samples = []
        for f in files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data["image_path"] = str(f.with_suffix(".jpg"))
                data["status"] = status
                samples.append(data)
            except Exception:
                continue
        return samples

    def review(self, sample_id: str, action: str) -> str:
        """Aplica a decisão do revisor e move a amostra.

        Ações:
            * ``correto``      – pré-rótulos corretos → aprovados;
            * ``falso_alarme`` – a pessoa estava com o EPI: no_X vira X → aprovados;
            * ``civil``        – pessoas do frame são civis/pedestres: Person vira civil e
                                 rótulos de EPI são removidos → aprovados;
            * ``descartar``    – frame inútil/ambíguo → rejeitados.

        Returns:
            Status de destino.
        """
        if action not in REVIEW_ACTIONS:
            raise ValueError(f"Ação de revisão inválida: {action}")
        if not _SAMPLE_ID_RE.match(sample_id or ""):  # impede caminhos como "../../arquivo"
            raise ValueError(f"Identificador de amostra inválido: {sample_id!r}")

        with self._lock:
            src = self.output_dir / STATUS_PENDING
            meta_path = src / f"{sample_id}.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Amostra pendente não encontrada: {sample_id}")

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            dets: list[dict[str, Any]] = meta.get("detections", [])

            if action == "falso_alarme":
                for d in dets:
                    d["class_name"] = FLIP_VIOLATION.get(d["class_name"], d["class_name"])
            elif action == "civil":
                dets = [{**d, "class_name": "civil"} for d in dets if d["class_name"] in ("Person", "civil")]

            meta["detections"] = dets
            meta["review"] = {"action": action, "at": datetime.now().isoformat(timespec="seconds")}

            dest_status = STATUS_REJECTED if action == "descartar" else STATUS_APPROVED
            dest = self.output_dir / dest_status
            dest.mkdir(parents=True, exist_ok=True)

            shutil.move(str(src / f"{sample_id}.jpg"), str(dest / f"{sample_id}.jpg"))
            txt_src = src / f"{sample_id}.txt"
            if txt_src.exists():
                txt_src.unlink()
            (dest / f"{sample_id}.txt").write_text(
                self._to_yolo(dets, meta.get("width", 1), meta.get("height", 1)), encoding="utf-8"
            )
            (dest / f"{sample_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            meta_path.unlink()

            if self._pending_count is not None:
                self._pending_count = max(0, self._pending_count - 1)
            return dest_status

    def stats(self) -> dict[str, int]:
        """Quantidade de amostras por status."""
        return {
            s: len(list((self.output_dir / s).glob("*.json"))) if (self.output_dir / s).exists() else 0
            for s in ALL_STATUSES
        }

    # ------------------------------------------------------------------
    # Exportação para treino
    # ------------------------------------------------------------------
    def export_dataset(self, dest_dir: str | Path = "datasets/epi_ativo", val_ratio: float = 0.2) -> dict[str, Any]:
        """Monta um dataset YOLO (train/val) a partir das amostras aprovadas.

        A divisão é feita por VÍDEO de origem (e não por frame) para evitar que
        frames quase iguais caiam em treino e validação ao mesmo tempo.
        """
        dest = Path(dest_dir)
        approved = self.list_samples(STATUS_APPROVED)
        if not approved:
            return {"train": 0, "val": 0, "data_yaml": None}

        # A pasta de destino é APAGADA antes da exportação: recusa destinos perigosos
        # (raiz do disco, pasta do projeto, ou a própria pasta de coleta com as amostras)
        dest_res = dest.resolve()
        out_res = self.output_dir.resolve()
        if dest_res == Path(dest_res.anchor) or dest_res == out_res or dest_res in out_res.parents:
            raise ValueError(f"Destino de exportação inseguro (seria apagado): {dest_res}")
        if dest.exists() and any(dest.iterdir()) and not (dest / "data.yaml").exists():
            raise ValueError(f"Destino '{dest_res}' não está vazio e não parece um dataset exportado anteriormente.")

        if dest.exists():
            shutil.rmtree(dest)
        for split in ("train", "val"):
            (dest / "images" / split).mkdir(parents=True, exist_ok=True)
            (dest / "labels" / split).mkdir(parents=True, exist_ok=True)

        def bucket(key: str) -> float:
            return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF

        splits = {s["id"]: ("val" if bucket(s.get("source") or s["id"]) < val_ratio else "train") for s in approved}
        # Com poucos vídeos, todos podem cair no mesmo lado: recorre à divisão por amostra
        if len(set(splits.values())) < 2 and len(approved) > 1:
            splits = {s["id"]: ("val" if bucket(s["id"]) < val_ratio else "train") for s in approved}
            if len(set(splits.values())) < 2:
                splits[approved[0]["id"]] = "val"

        counts = {"train": 0, "val": 0}
        src = self.output_dir / STATUS_APPROVED
        for s in approved:
            split = splits[s["id"]]
            shutil.copy2(src / f"{s['id']}.jpg", dest / "images" / split / f"{s['id']}.jpg")
            label_src = src / f"{s['id']}.txt"
            label_dst = dest / "labels" / split / f"{s['id']}.txt"
            if label_src.exists():
                shutil.copy2(label_src, label_dst)
            else:
                label_dst.write_text("", encoding="utf-8")
            counts[split] += 1

        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(TRAINING_CLASSES))
        data_yaml = dest / "data.yaml"
        data_yaml.write_text(
            f"# Dataset gerado pelo Aprendizado Ativo em {datetime.now():%Y-%m-%d %H:%M}\n"
            f"path: {dest.resolve().as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"names:\n{names}\n",
            encoding="utf-8",
        )
        return {**counts, "data_yaml": str(data_yaml.resolve())}

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------
    def _ensure_dirs(self) -> None:
        for s in ALL_STATUSES:
            (self.output_dir / s).mkdir(parents=True, exist_ok=True)
        classes_file = self.output_dir / "classes.txt"
        if not classes_file.exists():
            classes_file.write_text("\n".join(TRAINING_CLASSES) + "\n", encoding="utf-8")

    def _get_pending_count(self) -> int:
        if self._pending_count is None:
            folder = self.output_dir / STATUS_PENDING
            self._pending_count = len(list(folder.glob("*.json"))) if folder.exists() else 0
        return self._pending_count

    @staticmethod
    def _to_yolo(dets: list[dict[str, Any]], width: int, height: int) -> str:
        width = max(1, int(width))
        height = max(1, int(height))
        lines = []
        for d in dets:
            idx = CLASS_INDEX.get(d["class_name"])
            if idx is None:
                continue
            x1, y1, x2, y2 = d["bbox"]
            x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
            y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            cx = (x1 + bw / 2) / width
            cy = (y1 + bh / 2) / height
            lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
        return "\n".join(lines) + ("\n" if lines else "")

    @staticmethod
    def _dhash(frame: np.ndarray, size: int = 8) -> int:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
        diff = small[:, 1:] > small[:, :-1]
        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bit)
        return value

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
