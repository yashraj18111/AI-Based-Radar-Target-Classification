"""Data loading, preprocessing, and visualization for radar datasets."""

from preprocessing.loader import RadarDataset, RadarSample, create_dataset
from preprocessing.preprocess import PreprocessConfig, run_preprocessing
from preprocessing.splitter import SplitManifest, summarize_splits
from preprocessing.storage import load_processed_sample, load_manifest
from preprocessing.transforms import transform_sample
from preprocessing.validators import filter_valid_samples

__all__ = [
    "RadarDataset",
    "RadarSample",
    "create_dataset",
    "PreprocessConfig",
    "run_preprocessing",
    "SplitManifest",
    "summarize_splits",
    "load_processed_sample",
    "load_manifest",
    "transform_sample",
    "filter_valid_samples",
    "visualize_sample",
    "visualize_batch",
]


def __getattr__(name: str):
    if name in ("visualize_sample", "visualize_batch"):
        from preprocessing import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
