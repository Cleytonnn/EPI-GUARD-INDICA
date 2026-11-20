"""Pacote de interfaces e portas do domínio."""

from src.core.interfaces.alert_storage import IAlertStorage
from src.core.interfaces.detector import IDetector
from src.core.interfaces.notifier import INotifier
from src.core.interfaces.processed_repo import IProcessedVideosRepository
from src.core.interfaces.report_repo import IViolationReportRepository
from src.core.interfaces.video_reader import IVideoReader

__all__ = [
    "IDetector",
    "IVideoReader",
    "INotifier",
    "IAlertStorage",
    "IProcessedVideosRepository",
    "IViolationReportRepository",
]
