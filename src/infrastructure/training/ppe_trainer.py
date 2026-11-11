"""Pipeline de treinamento e avaliação do modelo de EPIs (metodologia do TCC).

Referência: "Detecção de EPI's com Ferramenta de Visão Computacional" (UFPA, 2024).

Metodologia reproduzida:
1. Base de dados anotada no formato YOLO (``.txt`` por imagem).
2. Divisão **83% treino / 12% validação / 5% teste** (cap. 4.0.3).
3. *Transfer learning* a partir de pesos YOLOv8 pré-treinados (cap. 4.0.4.2).
4. Hiperparâmetros: ``epochs``, ``workers=2``, ``batch`` ajustado por tamanho de
   modelo (n=80, s=60, m=40, l=20 no TCC; aqui limitado pela VRAM disponível) e
   *data augmentation* (``degrees``, ``scale``, ``mixup``, ``fliplr``).
5. Métricas (cap. 4.2): Precision, Recall, F1-score e mAP, por classe.
6. Comparativo entre as distribuições n/s/m (cap. 5.6).
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Proporção do TCC: 83% treino, 12% validação, 5% teste
TCC_SPLIT = (0.83, 0.12, 0.05)


@dataclass
class TrainingConfig:
    """Hiperparâmetros de treino (padrão = recomendações do TCC adaptadas)."""

    model: str = "yolov8s.pt"
    epochs: int = 100
    patience: int = 30
    imgsz: int = 640
    batch: int = 16
    workers: int = 2
    device: str = "0"
    seed: int = 42
    # Data augmentation (cap. 4.0.4.2 do TCC)
    degrees: float = 15.0
    scale: float = 0.4
    mixup: float = 0.1
    fliplr: float = 0.5
    mosaic: float = 1.0
    hsv_h: float = 0.015
    hsv_s: float = 0.6
    hsv_v: float = 0.4
    project: str = "runs/tcc"
    name: str = "epi"


@dataclass
class ClassMetrics:
    name: str
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float


@dataclass
class EvaluationReport:
    model: str
    split: str
    precision: float
    recall: float
    f1: float
    map50: float
    map50_95: float
    train_minutes: float = 0.0
    per_class: List[ClassMetrics] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"### {self.model} ({self.split})",
            "",
            f"- Precision: **{self.precision:.3f}** | Recall: **{self.recall:.3f}** | "
            f"F1: **{self.f1:.3f}** | mAP50: **{self.map50:.3f}** | mAP50-95: **{self.map50_95:.3f}**",
            f"- Tempo de treino: {self.train_minutes:.1f} min",
            "",
            "| Classe | Precision | Recall | F1 | mAP50 | mAP50-95 |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.per_class:
            lines.append(
                f"| {c.name} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {c.map50:.3f} | {c.map50_95:.3f} |"
            )
        return "\n".join(lines)


def f1_score(precision: float, recall: float) -> float:
    """F1 = 2·P·R / (P + R) — equação 4.3 do TCC."""
    denom = precision + recall
    return 0.0 if denom <= 0 else 2.0 * precision * recall / denom


def split_counts(total: int, ratios: tuple = TCC_SPLIT) -> tuple:
    """Calcula quantidades (train, val, test) que somam exatamente ``total``."""
    n_val = int(round(total * ratios[1]))
    n_test = int(round(total * ratios[2]))
    n_train = total - n_val - n_test
    return n_train, n_val, n_test


def _collect_pairs(source_dir: Path) -> List[tuple]:
    """Coleta pares (imagem, label) de um dataset YOLO com subpastas images/ e labels/."""
    pairs = []
    images_root = source_dir / "images"
    labels_root = source_dir / "labels"
    for img in sorted(images_root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = img.relative_to(images_root).with_suffix(".txt")
        label = labels_root / rel
        if label.exists():
            pairs.append((img, label))
    return pairs


def prepare_tcc_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    names: Dict[int, str],
    ratios: tuple = TCC_SPLIT,
    seed: int = 42,
    extra_sources: Optional[List[str | Path]] = None,
) -> Path:
    """Reagrupa um dataset YOLO na divisão 83/12/5 do TCC e gera o ``data.yaml``.

    Args:
        source_dir: Dataset YOLO original (images/{train,val,test}, labels/...).
        output_dir: Destino do dataset reorganizado.
        names: Mapeamento ``id -> nome`` das classes.
        ratios: Proporções (treino, validação, teste).
        seed: Semente para divisão reprodutível.
        extra_sources: Outros datasets YOLO (ex.: frames aprovados no Aprendizado Ativo).

    Returns:
        Caminho do ``data.yaml`` gerado.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    pairs = _collect_pairs(source_dir)
    for extra in extra_sources or []:
        pairs.extend(_collect_pairs(Path(extra)))
    if not pairs:
        raise FileNotFoundError(f"Nenhum par imagem/label encontrado em {source_dir}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_train, n_val, _ = split_counts(len(pairs), ratios)
    buckets = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split, items in buckets.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, label in items:
            shutil.copy2(img, output_dir / "images" / split / img.name)
            shutil.copy2(label, output_dir / "labels" / split / f"{img.stem}.txt")
        logger.info("   %-5s: %d imagens", split, len(items))

    data_yaml = output_dir / "data.yaml"
    data = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in names.items()},
    }
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml


def read_class_names(data_yaml: str | Path) -> Dict[int, str]:
    """Lê os nomes das classes de um ``data.yaml``."""
    raw = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = raw.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def train_model(data_yaml: str | Path, cfg: TrainingConfig) -> Path:
    """Executa o treinamento (transfer learning) e retorna o caminho do ``best.pt``."""
    from ultralytics import YOLO

    logger.info("🏋️ Treinando %s | epochs=%d batch=%d imgsz=%d", cfg.model, cfg.epochs, cfg.batch, cfg.imgsz)
    model = YOLO(cfg.model)
    results = model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        patience=cfg.patience,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        workers=cfg.workers,
        device=cfg.device,
        seed=cfg.seed,
        degrees=cfg.degrees,
        scale=cfg.scale,
        mixup=cfg.mixup,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        project=cfg.project,
        name=cfg.name,
        exist_ok=True,
        plots=True,
        amp=True,
        cos_lr=True,
        verbose=True,
    )
    save_dir = Path(getattr(results, "save_dir", Path(cfg.project) / cfg.name))
    return save_dir / "weights" / "best.pt"


def evaluate_model(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = 640,
    device: str = "0",
    train_minutes: float = 0.0,
) -> EvaluationReport:
    """Avalia um modelo e calcula Precision, Recall, F1 e mAP (global e por classe)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz, device=device, verbose=False, plots=True)
    box = metrics.box
    names = model.names

    per_class: List[ClassMetrics] = []
    for idx, cls_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(idx)
        per_class.append(ClassMetrics(
            name=names[int(cls_id)],
            precision=float(p),
            recall=float(r),
            f1=f1_score(float(p), float(r)),
            map50=float(ap50),
            map50_95=float(ap),
        ))

    return EvaluationReport(
        model=Path(str(weights)).parent.parent.name,
        split=split,
        precision=float(box.mp),
        recall=float(box.mr),
        f1=f1_score(float(box.mp), float(box.mr)),
        map50=float(box.map50),
        map50_95=float(box.map),
        train_minutes=train_minutes,
        per_class=per_class,
    )


def save_reports(reports: List[EvaluationReport], output_dir: str | Path) -> Path:
    """Salva o comparativo em JSON e Markdown (equivalente à Tabela 2 do TCC)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metricas.json").write_text(
        json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    header = [
        "# Comparativo de modelos YOLOv8 — Detecção de EPIs",
        "",
        "| Modelo | Split | Precision | Recall | F1 | mAP50 | mAP50-95 | Treino (min) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = [
        f"| {r.model} | {r.split} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
        f"{r.map50:.3f} | {r.map50_95:.3f} | {r.train_minutes:.1f} |"
        for r in reports
    ]
    details = ["", *[r.to_markdown() + "\n" for r in reports]]
    md = output_dir / "comparativo.md"
    md.write_text("\n".join(header + rows + details), encoding="utf-8")
    return md
