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


def test_compliance_service_infers_missing_helmet_on_person():
    """Garante que uma Pessoa sem capacete gere alerta de 'no_helmet' quando helmet for obrigatorio."""
    service = ComplianceService(required_ppe=["helmet"], min_violation_frames=2, infer_missing_ppe=True)
    person = Detection(class_name="Person", confidence=0.8, bbox=(100, 100, 300, 500), track_id=10)

    # Frame 1: pessoa sem capacete detectada (1º frame)
    v1 = service.check([person])
    assert len(v1) == 0

    # Frame 2: pessoa continua sem capacete (2º frame consecutivo) -> CONFIRMA VIOLAÇÃO
    v2 = service.check([person])
    assert len(v2) == 1
    assert v2[0].violation_type == "no_helmet"

    # Se colocarmos um capacete na pessoa no frame seguinte, não deve gerar violação
    service.reset()
    helmet = Detection(class_name="helmet", confidence=0.9, bbox=(120, 100, 280, 220), track_id=11)
    v_with_helmet = service.check([person, helmet])
    assert len(v_with_helmet) == 0


def test_compliance_service_infers_missing_gloves_on_person():
    """Garante que uma Pessoa sem luva gere alerta de 'no_gloves' quando gloves for obrigatorio."""
    service = ComplianceService(required_ppe=["gloves"], min_violation_frames=2, infer_missing_ppe=True)
    person = Detection(class_name="Person", confidence=0.8, bbox=(100, 100, 300, 500), track_id=20)

    # Frame 1: pessoa sem luvas detectada (1º frame)
    v1 = service.check([person])
    assert len(v1) == 0

    # Frame 2: pessoa continua sem luvas -> CONFIRMA VIOLAÇÃO NO_GLOVES
    v2 = service.check([person])
    assert len(v2) == 1
    assert v2[0].violation_type == "no_gloves"

    # Se a pessoa estiver usando luvas (na região das mãos/braços), nenhuma violação é gerada
    service.reset()
    gloves = Detection(class_name="gloves", confidence=0.85, bbox=(120, 320, 180, 380), track_id=21)
    v_with_gloves = service.check([person, gloves])
    assert len(v_with_gloves) == 0

