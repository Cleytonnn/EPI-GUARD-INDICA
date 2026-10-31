"""Testes unitários do coletor de Aprendizado Ativo."""

import json

import numpy as np
import pytest

from src.core.domain.detection import Detection
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation
from src.infrastructure.training.active_learning_collector import (
    CLASS_INDEX,
    ActiveLearningCollector,
)


def _frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)


def _meta(frame_idx: int, video: str = "CAM_A.mp4") -> VehicleMetadata:
    return VehicleMetadata(plate="ABC1D23", record_date="2026-10-30", video_name=video).with_time(
        frame_idx / 25.0, frame_idx
    )


@pytest.fixture
def collector(tmp_path):
    return ActiveLearningCollector(
        output_dir=tmp_path / "coleta",
        random_rate=0.0,
        min_frame_gap=10,
        seed=1,
    )


def test_ignores_confident_frames_without_alert(collector):
    dets = [Detection("Person", 0.95, (10, 10, 100, 200), 1), Detection("helmet", 0.9, (30, 10, 70, 40), 1)]
    assert collector.consider(_frame(1), dets, [], _meta(10)) is None
    assert collector.stats()["pendentes"] == 0


def test_collects_uncertain_frame_with_yolo_labels(collector):
    dets = [
        Detection("Person", 0.92, (10, 10, 110, 210), 1),
        Detection("no_helmet", 0.45, (40, 10, 80, 50), 1),
        Detection("vest", 0.45, (20, 60, 100, 140), 1),  # fora das classes de treino
    ]
    sample_id = collector.consider(_frame(2), dets, [], _meta(30))
    assert sample_id is not None

    pending = collector.output_dir / "pendentes"
    assert (pending / f"{sample_id}.jpg").exists()
    labels = (pending / f"{sample_id}.txt").read_text().strip().splitlines()
    assert len(labels) == 2
    assert labels[1].split()[0] == str(CLASS_INDEX["no_helmet"])

    meta = json.loads((pending / f"{sample_id}.json").read_text(encoding="utf-8"))
    assert meta["reasons"] == ["incerto"]
    assert meta["plate"] == "ABC1D23"


def test_respects_gap_and_deduplicates(collector):
    dets = [Detection("no_gloves", 0.4, (10, 10, 40, 40), 1)]
    assert collector.consider(_frame(3), dets, [], _meta(100)) is not None
    # Muito próximo no tempo
    assert collector.consider(_frame(4), dets, [], _meta(105)) is None
    # Mesmo conteúdo visual (duplicado), mesmo após o intervalo
    assert collector.consider(_frame(3), dets, [], _meta(200)) is None
    # Frame diferente e após o intervalo
    assert collector.consider(_frame(5), dets, [], _meta(300)) is not None


def test_alert_bypasses_gap(collector):
    dets = [Detection("no_helmet", 0.9, (10, 10, 40, 40), 1)]
    viol = Violation("no_helmet", 1, 0.0, 3, 0.9, (10, 10, 40, 40))
    assert collector.consider(_frame(6), [Detection("no_gloves", 0.4, (0, 0, 30, 30), 1)], [], _meta(10)) is not None
    sample_id = collector.consider(_frame(7), dets, [viol], _meta(12))
    assert sample_id is not None


def test_review_false_alarm_flips_violation_labels(collector):
    dets = [Detection("Person", 0.9, (10, 10, 110, 210), 1), Detection("no_helmet", 0.5, (40, 10, 80, 50), 1)]
    sample_id = collector.consider(_frame(8), dets, [], _meta(10))

    assert collector.review(sample_id, "falso_alarme") == "aprovados"
    approved = collector.output_dir / "aprovados"
    classes = [int(line.split()[0]) for line in (approved / f"{sample_id}.txt").read_text().splitlines()]
    assert CLASS_INDEX["helmet"] in classes
    assert CLASS_INDEX["no_helmet"] not in classes
    assert collector.stats() == {"pendentes": 0, "aprovados": 1, "rejeitados": 0}


def test_review_civil_and_discard(collector):
    dets = [Detection("Person", 0.9, (10, 10, 110, 210), 1), Detection("no_uniform_rf", 0.5, (20, 60, 100, 140), 1)]
    s1 = collector.consider(_frame(9), dets, [], _meta(10))
    s2 = collector.consider(_frame(10), dets, [], _meta(50))

    collector.review(s1, "civil")
    labels = (collector.output_dir / "aprovados" / f"{s1}.txt").read_text().splitlines()
    assert [int(line.split()[0]) for line in labels] == [CLASS_INDEX["civil"]]

    assert collector.review(s2, "descartar") == "rejeitados"
    with pytest.raises(ValueError):
        collector.review(s2, "acao_invalida")


def test_export_dataset_creates_train_val_split(collector, tmp_path):
    dets = [Detection("no_gloves", 0.4, (10, 10, 60, 60), 1)]
    ids = [collector.consider(_frame(20 + i), dets, [], _meta(100 * (i + 1), f"VID_{i}.mp4")) for i in range(6)]
    for sid in ids:
        collector.start_session("nova")  # garante que nenhuma regra de sessão interfira
        collector.review(sid, "correto")

    result = collector.export_dataset(tmp_path / "epi_ativo", val_ratio=0.3)
    assert result["train"] + result["val"] == 6
    assert result["train"] > 0 and result["val"] > 0
    assert (tmp_path / "epi_ativo" / "data.yaml").exists()


def test_disabled_collector_does_nothing(tmp_path):
    c = ActiveLearningCollector(output_dir=tmp_path / "off", enabled=False)
    dets = [Detection("no_helmet", 0.4, (10, 10, 40, 40), 1)]
    assert c.consider(_frame(1), dets, [], _meta(10)) is None
    assert not (tmp_path / "off").exists()

