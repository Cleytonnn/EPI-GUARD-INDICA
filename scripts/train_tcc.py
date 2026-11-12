"""CLI de treinamento seguindo a metodologia do TCC (YOLOv8 + transfer learning).

Exemplos:
    python scripts/train_tcc.py --models yolov8s.pt --epochs 100
    python scripts/train_tcc.py --models yolov8n.pt yolov8s.pt yolov8m.pt --epochs 100
    python scripts/train_tcc.py --models yolov8s.pt --use-active-learning
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.infrastructure.training.ppe_trainer import (  # noqa: E402
    CANONICAL_NAMES,
    TrainingConfig,
    evaluate_model,
    prepare_tcc_dataset,
    save_reports,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_tcc")

# Batch por tamanho de modelo (TCC: n=80, s=60, m=40, l=20 em GPU de 16GB; ajustado p/ 8GB)
DEFAULT_BATCH = {"n": 32, "s": 16, "m": 8, "l": 4}
DEFAULT_SOURCES = [
    "datasets/construction-ppe",
    "datasets/synapsis_ppe",
    "datasets/ppe_roboflow",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", default=None, help="Lista de datasets YOLO a unificar")
    p.add_argument("--output", default="datasets/epi_tcc", help="Dataset reorganizado 83/12/5")
    p.add_argument("--max-per-source", type=int, default=3500, help="Limite estratificado por fonte (0 = ilimitado)")
    p.add_argument("--models", nargs="+", default=["yolov8s.pt"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=0, help="0 = automático por tamanho do modelo")
    p.add_argument("--device", default="0")
    p.add_argument("--use-active-learning", action="store_true", help="Inclui frames aprovados do Aprendizado Ativo")
    p.add_argument("--keep-split", action="store_true", help="Reutiliza o dataset já reorganizado")
    p.add_argument("--resume", action="store_true", help="Retoma treinos interrompidos (mesmo dataset) de onde pararam")
    args = p.parse_args()

    candidates = args.sources or DEFAULT_SOURCES
    sources = [ROOT / s for s in candidates if (ROOT / s).exists()]
    if args.use_active_learning:
        approved = ROOT / "datasets" / "coleta_ativa" / "aprovados"
        if approved.exists():
            sources.append(approved)
    if not sources:
        raise FileNotFoundError(f"Nenhum dataset encontrado em: {candidates}")

    output = ROOT / args.output
    data_yaml = output / "data.yaml"
    if not (args.keep_split and data_yaml.exists()):
        logger.info("📂 Unificando %d fonte(s) na divisão TCC 83%% treino / 12%% val / 5%% teste ...", len(sources))
        data_yaml = prepare_tcc_dataset(
            sources[0],
            output,
            names=CANONICAL_NAMES,
            extra_sources=sources[1:],
            max_per_source=args.max_per_source,
        )

    reports = []
    best_overall = None
    for model_name in args.models:
        size_letter = Path(model_name).stem.replace("yolov8", "").replace("yolo11", "")[:1] or "s"
        cfg = TrainingConfig(
            model=model_name,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch or DEFAULT_BATCH.get(size_letter, 16),
            device=args.device,
            name=f"epi_{Path(model_name).stem}",
        )
        t0 = time.time()
        best = train_model(data_yaml, cfg, resume=args.resume)
        minutes = (time.time() - t0) / 60.0
        report = evaluate_model(best, data_yaml, split="test", imgsz=args.imgsz, device=args.device, train_minutes=minutes)
        reports.append(report)
        logger.info("📊 %s -> P=%.3f R=%.3f F1=%.3f mAP50=%.3f", report.model, report.precision,
                    report.recall, report.f1, report.map50)
        if best_overall is None or report.map50 > best_overall[1]:
            best_overall = (best, report.map50)

    md = save_reports(reports, ROOT / "runs" / "tcc")
    logger.info("📝 Relatório: %s", md)

    if best_overall:
        dest = ROOT / "models" / "epi_best.pt"
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(best_overall[0], dest)
        logger.info("✅ Melhor modelo (mAP50=%.3f) copiado para %s", best_overall[1], dest)


if __name__ == "__main__":
    main()
