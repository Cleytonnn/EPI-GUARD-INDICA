"""Pacote de domínio."""

from src.core.domain.detection import BoundingBox, Detection
from src.core.domain.ppe_type import PPEClass, ViolationClass
from src.core.domain.vehicle_metadata import VehicleMetadata
from src.core.domain.violation import Violation

__all__ = [
    "PPEClass",
    "ViolationClass",
    "Detection",
    "BoundingBox",
    "Violation",
    "VehicleMetadata",
]
