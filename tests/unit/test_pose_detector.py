"""Testes unitários para o PoseEPIDetector."""

import numpy as np
import pytest
from src.infrastructure.ai.pose_detector import PoseEPIDetector


def test_pose_detector_initialization():
    detector = PoseEPIDetector(model_path="yolov8n-pose.pt", confidence=0.25)
    assert detector.model is not None
    assert detector.confidence == 0.25
    assert detector.device in {"cuda", "cpu"}


def test_pose_detector_empty_frame():
    detector = PoseEPIDetector(model_path="yolov8n-pose.pt")
    assert detector.detect(None) == []
    assert detector.detect(np.array([])) == []


def test_pose_detector_inference_on_blank_image():
    detector = PoseEPIDetector(model_path="yolov8n-pose.pt", track=False)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_frame)
    # Em uma imagem toda preta, não deve haver detecções falsas de pessoas
    assert isinstance(detections, list)
    assert len(detections) == 0


def _make_keypoints(sl=(200, 180), sr=(260, 180), hl=(205, 320), hr=(255, 320), wrist_l=(195, 280), elbow_l=(195, 230)):
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[5] = [sl[0], sl[1], 0.9]
    kps[6] = [sr[0], sr[1], 0.9]
    kps[7] = [elbow_l[0], elbow_l[1], 0.9]
    kps[9] = [wrist_l[0], wrist_l[1], 0.9]
    kps[11] = [hl[0], hl[1], 0.9]
    kps[12] = [hr[0], hr[1], 0.9]
    return kps


def test_pose_detector_detects_shaded_dark_green_helmet():
    detector = PoseEPIDetector.__new__(PoseEPIDetector)
    detector.helmet_threshold = 0.22
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Capacete verde escuro na sombra (H~68, S~115, V~34 em OpenCV HSV -> BGR aproximado (20, 34, 18))
    frame[90:160, 195:265] = (20, 34, 18)
    kps = _make_keypoints()
    has_helmet, score, _ = detector._analyze_helmet(frame, (170, 95, 290, 420), kps)
    assert has_helmet is True
    assert score >= 0.22


def test_pose_detector_detects_side_profile_beige_orange_uniform_rf():
    detector = PoseEPIDetector.__new__(PoseEPIDetector)
    detector.uniform_rf_threshold = 0.12
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Pessoa de perfil: ombros colapsados em X (225 e 232), corpo cáqui + pala laranja superior + faixa refletiva
    frame[145:195, 195:270] = (25, 115, 230)   # Laranja fluorescente na pala
    frame[195:330, 195:270] = (115, 150, 175)  # Cáqui/bege RF
    frame[240:255, 195:270] = (185, 185, 185)  # Faixa refletiva cinza/prata
    kps = _make_keypoints(sl=(225, 180), sr=(232, 180), hl=(222, 320), hr=(235, 320))
    has_uniform, score, _ = detector._analyze_uniform_rf(frame, (185, 95, 280, 440), kps)
    assert has_uniform is True
    assert score >= 0.60


def test_pose_detector_gloves_vaqueta_vs_occluded_and_partial_bg():
    detector = PoseEPIDetector.__new__(PoseEPIDetector)
    detector.gloves_skin_threshold = 0.35

    # 1. Luva de vaqueta (couro claro/cinza-esbranquiçado) -> True
    frame_glv = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_glv[60:140, 60:140] = (195, 195, 200)
    has_glv, score, _ = detector._analyze_gloves(frame_glv, (100, 100), (100, 70), 80)
    assert has_glv is True and score >= 0.55

    # 2. Mão oclusa na manga laranja RF -> None (nem True nem False)
    frame_occ = np.zeros((200, 200, 3), dtype=np.uint8)
    frame_occ[60:140, 60:140] = (15, 125, 235)
    has_occ, _, _ = detector._analyze_gloves(frame_occ, (100, 100), (100, 70), 80)
    assert has_occ is None

    # 3. Fundo escuro com pequena faixa clara (< 34%) -> None (evita falso positivo de parede branca)
    frame_bg = np.full((200, 200, 3), (30, 30, 30), dtype=np.uint8)
    frame_bg[85:100, 85:130] = (220, 220, 220)
    has_bg, _, _ = detector._analyze_gloves(frame_bg, (100, 100), (100, 70), 80)
    assert has_bg is None


def test_pose_detector_does_not_promote_street_pedestrian_as_cabin_worker():
    class _StubBoxes:
        def __init__(self):
            import torch
            self.xyxy = torch.tensor([[100.0, 560.0, 290.0, 1030.0]])
            self.conf = torch.tensor([0.88])
            self.id = torch.tensor([13])

        def __len__(self):
            return 1

    class _StubKeypoints:
        def __init__(self):
            import torch
            kps = _make_keypoints(sl=(160, 660), sr=(230, 660), hl=(165, 810), hr=(225, 810), wrist_l=(155, 760), elbow_l=(155, 710))
            self.data = torch.tensor(kps).unsqueeze(0)

    class _StubResult:
        def __init__(self):
            self.boxes = _StubBoxes()
            self.keypoints = _StubKeypoints()

    class _StubModel:
        def __call__(self, **kwargs):
            return [_StubResult()]

    detector = PoseEPIDetector.__new__(PoseEPIDetector)
    detector.model = _StubModel()
    detector._confidence = 0.22
    detector.iou_threshold = 0.45
    detector.imgsz = 640
    detector.device = "cpu"
    detector.track = False
    detector.helmet_threshold = 0.22
    detector.uniform_rf_threshold = 0.12
    detector.gloves_skin_threshold = 0.35
    detector.only_company_workers = True
    detector.is_cabin = False

    # Em uma câmera externa com pedestre de camiseta preta, mantém como civil
    frame = np.full((1080, 1920, 3), (35, 35, 35), dtype=np.uint8)
    dets = detector.detect(frame)
    classes = [d.class_name for d in dets]
    assert "civil" in classes
    assert "Person" not in classes
    assert "no_helmet" not in classes


def test_pose_detector_ignores_civilian_yellow_dress_or_maroon_shirt():
    detector = PoseEPIDetector.__new__(PoseEPIDetector)
    detector.uniform_rf_threshold = 0.12
    kps = _make_keypoints(sl=(200, 180), sr=(260, 180), hl=(205, 320), hr=(255, 320))

    # Camiseta vinho/marrom fosco medida do pedestre real (S~107, V~131 -> BGR (76, 86, 131))
    frame_maroon = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_maroon[145:330, 185:275] = (76, 86, 131)
    has_u_maroon, _, _ = detector._analyze_uniform_rf(frame_maroon, (170, 95, 290, 440), kps)
    assert has_u_maroon is False

    # Vestido bege/amarelo claro sem pala laranja fluorescente
    frame_dress = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_dress[145:330, 185:275] = (140, 185, 200)
    has_u_dress, _, _ = detector._analyze_uniform_rf(frame_dress, (170, 95, 290, 440), kps)
    assert has_u_dress is False



