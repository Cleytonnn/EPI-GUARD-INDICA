"""Pacote de Casos de Uso da Aplicação."""

from src.core.use_cases.analyze_frame import AnalyzeFrameUseCase, FrameAnalysisResult
from src.core.use_cases.process_batch import ProcessBatchUseCase
from src.core.use_cases.process_stream import ProcessLiveStreamUseCase
from src.core.use_cases.process_video_file import ProcessVideoFileUseCase, VideoProcessingResult
from src.core.use_cases.watch_directory import WatchDirectoryUseCase

__all__ = [
    "ProcessVideoFileUseCase",
    "VideoProcessingResult",
    "ProcessBatchUseCase",
    "WatchDirectoryUseCase",
    "AnalyzeFrameUseCase",
    "FrameAnalysisResult",
    "ProcessLiveStreamUseCase",
]
