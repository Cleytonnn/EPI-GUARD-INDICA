"""Benchmark de estabilidade: compara detecção Pose (heurística) vs Híbrida em um vídeo.

Métricas por (track, EPI): número de trocas de estado presente<->ausente (flicker) e
fração de frames em que a ausência foi declarada.

Uso:
    python scripts/benchmark_stability.py gravacoes/ABC1D23/2026-10-04/camera_operacao_01.mp4 --frames 150
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import AppSettings  # noqa: E402
from src.container import Container  # noqa: E402
from src.infrastructure.video.universal_reader import UniversalVideoReader  # noqa: E402

TRACKED = {
    "helmet": "helmet",
    "no_helmet": "helmet",
    "gloves": "gloves",
    "no_gloves": "gloves",
    "vest": "vest",
    "no_vest": "vest",
    "uniform_rf": "uniform_rf",
    "no_uniform_rf": "uniform_rf",
}


def run(video: str, backend: str, frames: int, step: int) -> dict:
    settings = AppSettings.load()
    settings.detection.backend = backend
    detector = Container(settings).detector
    reader = UniversalVideoReader(video)
    history = defaultdict(list)  # (track, item) -> [0/1 ausente]
    n = 0
    read = 0
    while n < frames:
        ok, frame = reader.read()
        if not ok:
            break
        read += 1
        if read % step:
            continue
        n += 1
        for det in detector.detect(frame):
            item = TRACKED.get(det.class_name)
            if item is None or det.track_id in (None, -1):
                continue
            history[(det.track_id, item)].append(1 if det.class_name.startswith("no_") else 0)
    reader.release()

    flips = sum(sum(1 for a, b in zip(h, h[1:], strict=False) if a != b) for h in history.values())
    samples = sum(len(h) for h in history.values())
    absent = sum(sum(h) for h in history.values())
    return {
        "backend": backend,
        "frames": n,
        "series": len(history),
        "flips": flips,
        "flips_per_100": round(100.0 * flips / max(1, samples), 2),
        "absent_ratio": round(absent / max(1, samples), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--step", type=int, default=2, help="Analisa 1 a cada N frames")
    args = p.parse_args()
    for backend in ("pose", "hybrid"):
        print(run(args.video, backend, args.frames, args.step))


if __name__ == "__main__":
    main()
