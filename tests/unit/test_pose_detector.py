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

