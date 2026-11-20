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

import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Proporção do TCC: 83% treino, 12% validação, 5% teste
TCC_SPLIT = (0.83, 0.12, 0.05)

# Taxonomia unificada do projeto (compatível com ComplianceService e HybridEPIDetector)
CANONICAL_NAMES: dict[int, str] = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "Person",
    6: "no_helmet",
    7: "no_goggle",
    8: "no_gloves",
    9: "no_boots",
    10: "no_vest",
    11: "ear_protection",
    12: "mask",
}

CLASS_ALIASES: dict[str, str] = {
    "helmet": "helmet",
    "capacete": "helmet",
    "gloves": "gloves",
    "glove": "gloves",
    "luvas": "gloves",
    "vest": "vest",
    "colete": "vest",
    "uniform_rf": "vest",
    "coverall": "vest",
    "worker_rf": "vest",
    "boots": "boots",
    "safety_shoes": "boots",
    "safety-shoes": "boots",
    "botas": "boots",
    "goggles": "goggles",
    "glasses": "goggles",
    "safety_glasses": "goggles",
    "oculos": "goggles",
    "óculos": "goggles",
    "person": "Person",
    "civil": "Person",
    "no_helmet": "no_helmet",
    "no-helmet": "no_helmet",
    "no helmet": "no_helmet",
    "no_goggle": "no_goggle",
    "no_goggles": "no_goggle",
    "no-goggles": "no_goggle",
    "no goggles": "no_goggle",
    "no_safety_glasses": "no_goggle",
    "no-safety-glasses": "no_goggle",
    "no_gloves": "no_gloves",
    "no-gloves": "no_gloves",
    "no_glove": "no_gloves",
    "no-glove": "no_gloves",
    "no glove": "no_gloves",
    "no gloves": "no_gloves",
    "no_boots": "no_boots",
    "no-boots": "no_boots",
    "no boots": "no_boots",
    "nosafety_shoes": "no_boots",
    "no_safety_shoes": "no_boots",
    "no_vest": "no_vest",
    "no-vest": "no_vest",
    "no vest": "no_vest",
    "no_uniform_rf": "no_vest",
    "no_coverall": "no_vest",
    "no-coverall": "no_vest",
    "ear_protection": "ear_protection",
    "ear-protection": "ear_protection",
    "abafador": "ear_protection",
    "respirator_mask": "mask",
    "mask": "mask",
    "mascara": "mask",
    "máscara": "mask",
}


@dataclass
class TrainingConfig:
    """Hiperparâmetros de treino (alinhados ao cap. 4.0.4.2 do TCC)."""

    model: str = "yolov8s.pt"
    epochs: int = 100
    patience: int = 25
    imgsz: int = 640
    batch: int = 16
    workers: int = 2
    device: str = "0"
    seed: int = 42
    # Data augmentation (cap. 4.0.4.2 do TCC: degrees=30, scale=0.4, mixup=0.3, fliplr=0.5)
    degrees: float = 30.0
    scale: float = 0.4
    mixup: float = 0.3
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
    per_class: list[ClassMetrics] = field(default_factory=list)

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


def _canonical_id_map(
    source_names: dict[int, str],
    target_names: dict[int, str],
) -> dict[int, int]:
    """Mapeia IDs de classe de um dataset para os IDs canônicos de destino."""
    target_by_name = {name: idx for idx, name in target_names.items()}
    mapping: dict[int, int] = {}
    for src_id, raw_name in source_names.items():
        norm = CLASS_ALIASES.get(str(raw_name).strip().lower(), str(raw_name).strip())
        if norm in target_by_name:
            mapping[int(src_id)] = target_by_name[norm]
    return mapping


def remap_label_lines(text: str, id_map: dict[int, int]) -> list[str]:
    """Remapeia IDs de classes em um arquivo .txt YOLO, descartando classes não mapeadas."""
    out: list[str] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_id = int(float(parts[0]))
        except ValueError:
            continue
        if src_id not in id_map:
            continue
        out.append(f"{id_map[src_id]} " + " ".join(parts[1:]))
    return out


def _collect_pairs(source_dir: Path) -> list[tuple]:
    """Coleta pares (imagem, label) nos layouts Ultralytics (images/split) ou Roboflow (split/images)."""
    pairs: list[tuple] = []
    seen: set[Path] = set()

    def scan(img_dir: Path, lbl_dir: Path) -> None:
        if not img_dir.exists() or not lbl_dir.exists():
            return
        for img in sorted(img_dir.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXTENSIONS or img in seen:
                continue
            rel = img.relative_to(img_dir).with_suffix(".txt")
            label = lbl_dir / rel
            if label.exists():
                seen.add(img)
                pairs.append((img, label))

    scan(source_dir / "images", source_dir / "labels")
    for split_dir in ("train", "valid", "val", "test"):
        scan(source_dir / split_dir / "images", source_dir / split_dir / "labels")
    return pairs


def _sample_balanced_items(
    items: list[tuple],
    max_items: int,
    seed: int,
) -> list[tuple]:
    """Seleciona até ``max_items`` priorizando imagens com classes mais raras (ex.: no_gloves, vest, boots)."""
    if max_items <= 0 or len(items) <= max_items:
        return items

    from collections import Counter

    rng = random.Random(seed)
    class_freq: Counter[int] = Counter()
    parsed: list[tuple] = []
    for img, lines in items:
        classes = {int(ln.split()[0]) for ln in lines}
        for c in classes:
            class_freq[c] += 1
        parsed.append((img, lines, classes))

    # Pontua cada imagem pela raridade máxima das suas classes + ruído aleatório leve
    scored = []
    for img, lines, classes in parsed:
        rarity = max((1.0 / max(1, class_freq[c]) for c in classes), default=0.0)
        scored.append((rarity + rng.uniform(0.0, 1e-4), img, lines))
    scored.sort(key=lambda row: row[0], reverse=True)
    return [(img, lines) for _, img, lines in scored[:max_items]]


def prepare_tcc_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    names: dict[int, str] | None = None,
    ratios: tuple = TCC_SPLIT,
    seed: int = 42,
    extra_sources: list[str | Path] | None = None,
    max_per_source: int = 0,
) -> Path:
    """Reagrupa um ou mais datasets YOLO na divisão 83/12/5 do TCC e gera o ``data.yaml``.

    Unifica layouts Ultralytics e Roboflow e remapeia classes via ``CANONICAL_NAMES``
    quando a fonte possui ``data.yaml`` próprio.
    """
    target_names = names or CANONICAL_NAMES
    source_list = [Path(source_dir), *(Path(s) for s in (extra_sources or []))]
    output_dir = Path(output_dir)

    all_samples: list[tuple] = []  # (img_path, remapped_lines)
    for src_idx, src in enumerate(source_list):
        if not src.exists():
            continue
        yaml_path = src / "data.yaml"
        if yaml_path.exists():
            id_map = _canonical_id_map(read_class_names(yaml_path), target_names)
        else:
            id_map = {int(k): int(k) for k in target_names}

        src_items: list[tuple] = []
        for img, label in _collect_pairs(src):
            lines = remap_label_lines(label.read_text(encoding="utf-8", errors="ignore"), id_map)
            if lines:
                src_items.append((img, lines))

        if max_per_source > 0 and len(src_items) > max_per_source:
            src_items = _sample_balanced_items(src_items, max_per_source, seed + src_idx)

        logger.info("   Fonte %-20s: %d imagens válidas", src.name, len(src_items))
        all_samples.extend(src_items)

    if not all_samples:
        raise FileNotFoundError(f"Nenhum par imagem/label válido encontrado em {source_dir}")

    rng = random.Random(seed)
    rng.shuffle(all_samples)
    n_train, n_val, _ = split_counts(len(all_samples), ratios)
    buckets = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train : n_train + n_val],
        "test": all_samples[n_train + n_val :],
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    counter = 0
    for split, items in buckets.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lines in items:
            counter += 1
            stem = f"{counter:06d}_{img.stem[:40]}"
            shutil.copy2(img, output_dir / "images" / split / f"{stem}{img.suffix.lower()}")
            (output_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("   %-5s: %d imagens", split, len(items))

    data_yaml = output_dir / "data.yaml"
    data = {
        "path": str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in target_names.items()},
    }
    data_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml


def read_class_names(data_yaml: str | Path) -> dict[int, str]:
    """Lê os nomes das classes de um ``data.yaml``."""
    raw = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = raw.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): v for k, v in names.items()}


def find_resumable_checkpoint(cfg: TrainingConfig, data_yaml: str | Path) -> Path | None:
    """Retorna um checkpoint de execução interrompida (mesmos dados e não concluída).

    Testa ``last.pt`` e, se estiver corrompido (ex.: queda de energia/servidor durante a
    gravação), cai para os checkpoints periódicos ``epoch*.pt`` do mais recente ao mais antigo.
    O Ultralytics grava ``epoch=-1`` quando o treino termina normalmente; qualquer outro
    valor indica interrupção. Também confere o ``data.yaml`` para nunca retomar treino de
    outro dataset.
    """
    import torch

    weights_dirs = [
        Path("runs/detect") / cfg.project / cfg.name / "weights",
        Path(cfg.project) / cfg.name / "weights",
    ]
    candidates: list[Path] = []
    for wdir in weights_dirs:
        if wdir.exists():
            candidates.append(wdir / "last.pt")
            candidates.extend(sorted(wdir.glob("epoch*.pt"), key=lambda p: p.stat().st_mtime, reverse=True))

    for ckpt_path in candidates:
        if not ckpt_path.exists():
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if int(ckpt.get("epoch", -1)) < 0:
                return None  # treino já concluído
            trained_data = Path(str(ckpt.get("train_args", {}).get("data", ""))).resolve()
            if trained_data != Path(data_yaml).resolve():
                return None  # checkpoint de outro dataset
            return ckpt_path
        except Exception as exc:
            logger.warning("Checkpoint %s ilegível (%s); tentando o anterior.", ckpt_path.name, str(exc)[:60])
    return None


def train_model(data_yaml: str | Path, cfg: TrainingConfig, resume: bool = False) -> Path:
    """Executa o treinamento (transfer learning) e retorna o caminho do ``best.pt``.

    Com ``resume=True`` e um checkpoint interrompido compatível, continua da última época.
    """
    from ultralytics import YOLO

    checkpoint = find_resumable_checkpoint(cfg, data_yaml) if resume else None
    if checkpoint is not None:
        logger.info("⏯️ Retomando treino interrompido de %s", checkpoint)
        results = YOLO(str(checkpoint)).train(resume=True)
        save_dir = Path(getattr(results, "save_dir", checkpoint.parent.parent))
        return save_dir / "weights" / "best.pt"

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
        save_period=5,  # checkpoints epochN.pt: protegem contra last.pt corrompido numa queda
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

    per_class: list[ClassMetrics] = []
    for idx, cls_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(idx)
        per_class.append(
            ClassMetrics(
                name=names[int(cls_id)],
                precision=float(p),
                recall=float(r),
                f1=f1_score(float(p), float(r)),
                map50=float(ap50),
                map50_95=float(ap),
            )
        )

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


def save_reports(reports: list[EvaluationReport], output_dir: str | Path) -> Path:
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
