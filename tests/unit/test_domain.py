"""Testes unitários das entidades de domínio."""

from src.core.domain.detection import BoundingBox, Detection
from src.core.domain.violation import Violation


def test_detection_properties():
    det_helmet = Detection(class_name="helmet", confidence=0.88, bbox=(10, 20, 100, 200), track_id=1)
    assert not det_helmet.is_violation
    assert det_helmet.ppe_type == "helmet"
    assert "Capacete" in det_helmet.label_pt

    det_no_helmet = Detection(class_name="no_helmet", confidence=0.92, bbox=(10, 20, 100, 200), track_id=2)
    assert det_no_helmet.is_violation
    assert det_no_helmet.ppe_type == "helmet"
    assert "Sem Capacete" in det_no_helmet.label_pt


def test_bounding_box_calculations():
    bbox = BoundingBox(x1=10, y1=20, x2=110, y2=220)
    assert bbox.width == 100
    assert bbox.height == 200
    assert bbox.area == 20000
    assert bbox.tuple == (10, 20, 110, 220)


def test_violation_unique_key_generation():
    v1 = Violation(
        violation_type="no_gloves",
        track_id=5,
        timestamp=1000.0,
        consecutive_frames=4,
        confidence=0.85,
        bbox=(0, 0, 50, 50),
    )
    assert v1.unique_key == "no_gloves_track_5"
    assert "Sem Luvas" in v1.description
