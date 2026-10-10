"""Testes unitários do ComplianceService."""

from src.core.domain.detection import Detection
from src.core.services.compliance_service import ComplianceService


def test_compliance_service_consecutive_frames_logic():
    service = ComplianceService(min_violation_frames=3)
    det = Detection(class_name="no_helmet", confidence=0.9, bbox=(10, 10, 50, 50), track_id=1)

    # Frame 1: contador = 1, não confirma
    v1 = service.check([det])
    assert len(v1) == 0

    # Frame 2: contador = 2, não confirma
    v2 = service.check([det])
    assert len(v2) == 0

    # Frame 3: contador = 3, CONFIRMA
    v3 = service.check([det])
    assert len(v3) == 1
    assert v3[0].violation_type == "no_helmet"
    assert v3[0].consecutive_frames == 3

    # Frame 4: continua presente, mas já foi confirmada (não duplica alerta)
    v4 = service.check([det])
    assert len(v4) == 0


def test_compliance_service_reset():
    service = ComplianceService(min_violation_frames=2)
    det = Detection(class_name="no_gloves", confidence=0.8, bbox=(0, 0, 10, 10), track_id=2)

    service.check([det])
    service.reset()

    # Após reset, precisa contar do zero novamente
    assert len(service.check([det])) == 0
    assert len(service.check([det])) == 1
