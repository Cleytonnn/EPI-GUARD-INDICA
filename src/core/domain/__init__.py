"""Pacote de domínio."""

from src.core.domain.ppe_type import PPEClass, ViolationClass
from src.core.domain.detection import Detection, BoundingBox
from src.core.domain.violation import Violation
from src.core.domain.vehicle_metadata import VehicleMetadata

__all__ = [
    "PPEClass",
    "ViolationClass",
    "Detection",
    "BoundingBox",
    "Violation",
    "VehicleMetadata",
]
