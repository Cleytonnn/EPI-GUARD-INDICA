"""Testes da suavização temporal, fusão de evidências e detector híbrido."""

import numpy as np
import pytest

from src.core.domain.detection import Detection
from src.core.services.ppe_state_smoother import PPEState, PPEStateSmoother
from src.infrastructure.ai.hybrid_detector import HybridEPIDetector, fuse_evidence
from src.infrastructure.training.ppe_trainer import f1_score, split_counts


# ----------------------------------------------------------------- smoother
def test_smoother_starts_unknown_until_min_observations():
    s = PPEStateSmoother(min_observations=3)
    s.begin_frame()
    assert s.update("k", 0.9)[0] is PPEState.UNKNOWN
    assert s.update("k", 0.9)[0] is PPEState.UNKNOWN
    assert s.update("k", 0.9)[0] is PPEState.PRESENT


def test_smoother_ignores_isolated_flicker():
    """Um único frame 'ausente' no meio de vários 'presente' não vira infração."""
    s = PPEStateSmoother(alpha=0.35, min_observations=3)
    s.begin_frame()
    for _ in range(6):
        s.update("k", 0.9)
    state, _, _ = s.update("k", 0.0)
    assert state is PPEState.PRESENT


def test_smoother_transitions_to_absent_with_sustained_evidence():
    s = PPEStateSmoother(alpha=0.35, min_observations=3)
    s.begin_frame()
    for _ in range(4):
        s.update("k", 0.9)
    state = PPEState.PRESENT
    for _ in range(10):
        state, _, _ = s.update("k", 0.1)
    assert state is PPEState.ABSENT


def test_smoother_hysteresis_keeps_state_in_gray_zone():
    s = PPEStateSmoother(alpha=1.0, present_on=0.55, absent_on=0.33, min_observations=1)
    s.begin_frame()
    assert s.update("k", 0.9)[0] is PPEState.PRESENT
    assert s.update("k", 0.45)[0] is PPEState.PRESENT  # zona cinza mantém estado


def test_smoother_smooths_boxes_and_prunes_stale_keys():
    s = PPEStateSmoother(box_alpha=0.5, max_age=2)
    s.begin_frame()
    s.update("k", 0.9, (0, 0, 100, 100))
    _, _, box = s.update("k", 0.9, (20, 20, 120, 120))
    assert box == (10, 10, 110, 110)
    for _ in range(4):
        s.begin_frame()
    assert s.update("k", 0.9)[0] is PPEState.UNKNOWN  # chave expirou e recomeçou


def test_smoother_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        PPEStateSmoother(present_on=0.3, absent_on=0.5)


# ------------------------------------------------------------------- fusão
def test_fuse_model_positive_overrides_heuristic_absence():
    assert fuse_evidence(0.8, 0.0, False, True) > 0.55


def test_fuse_model_negative_declares_absence():
    assert fuse_evidence(0.0, 0.8, True, True) < 0.33


def test_fuse_heuristic_absence_alone_is_weak_for_items_with_negative_class():
    assert 0.25 <= fuse_evidence(0.0, 0.0, False, True) <= 0.33


# --------------------------------------------------------------- híbrido
class _FakePose:
    confidence = 0.25
    class_names = {}

    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame, skip_frames=1):
        return list(self.dets)

    def reset_tracking(self):
        pass


class _FakeBoxes:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    @property
    def xyxy(self):
        return _T(np.array([i[0] for i in self._items], dtype=float))

    @property
    def conf(self):
        return _T(np.array([i[1] for i in self._items], dtype=float))

    @property
    def cls(self):
        return _T(np.array([i[2] for i in self._items], dtype=float))


class _T:
    def __init__(self, arr):
        self.arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeResult:
    def __init__(self, items):
        self.boxes = _FakeBoxes(items) if items else None


class _FakePPEModel:
    names = {0: "helmet", 1: "gloves", 2: "vest", 7: "no_helmet", 9: "no_gloves"}

    def __init__(self, items_per_crop):
        self.items = items_per_crop

    def predict(self, source, **kwargs):
        return [_FakeResult(self.items) for _ in source]


def _pose_worker_without_helmet_heuristic():
    return [
        Detection("Person", 0.9, (100, 50, 300, 450), 1),
        Detection("no_helmet", 0.8, (150, 50, 250, 130), 1),  # heurística de cor errou
        Detection("no_uniform_rf", 0.85, (110, 120, 290, 300), 1),
    ]


def test_hybrid_model_helmet_prevents_false_violation():
    # Capacete visto pelo modelo (coords do recorte) -> nunca emite no_helmet
    crop_helmet = ([40, 10, 140, 90], 0.9, 0)
    det = HybridEPIDetector(
        pose_detector=_FakePose(_pose_worker_without_helmet_heuristic()),
        ppe_model=_FakePPEModel([crop_helmet]),
        smoother=PPEStateSmoother(min_observations=2),
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    names = []
    for _ in range(5):
        names = [d.class_name for d in det.detect(frame)]
    assert "helmet" in names
    assert "no_helmet" not in names


def test_hybrid_reports_no_helmet_when_model_sees_bare_head():
    crop_bare = ([40, 10, 140, 90], 0.85, 7)
    det = HybridEPIDetector(
        pose_detector=_FakePose(_pose_worker_without_helmet_heuristic()),
        ppe_model=_FakePPEModel([crop_bare]),
        smoother=PPEStateSmoother(min_observations=2),
    )
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    names = []
    for _ in range(6):
        names = [d.class_name for d in det.detect(frame)]
    assert "no_helmet" in names


def test_hybrid_emits_nothing_for_civil_without_ppe():
    civil = [Detection("civil", 0.9, (100, 50, 300, 450), 3)]
    det = HybridEPIDetector(
        pose_detector=_FakePose(civil),
        ppe_model=_FakePPEModel([]),
    )
    out = det.detect(np.zeros((600, 800, 3), dtype=np.uint8))
    assert [d.class_name for d in out] == ["civil"]


# --------------------------------------------------------------- treinamento
def test_split_counts_follow_tcc_ratio():
    train, val, test = split_counts(1416)
    assert train + val + test == 1416
    assert abs(train / 1416 - 0.83) < 0.01
    assert abs(val / 1416 - 0.12) < 0.01


def test_f1_score_matches_tcc_example():
    assert f1_score(0.85, 0.80) == pytest.approx(0.8242, abs=1e-3)
    assert f1_score(0.0, 0.0) == 0.0
