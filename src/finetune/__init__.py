"""Fine-tuning utilities for the existing RobustLens detector.

The package is deliberately separate from the inference pipeline.  Importing
it does not load the 740M-parameter model or contact a model hub.
"""

from .dataset import DatasetSummary, LocalEditDataset, discover_dataset, verify_split_groups
from .model import FineTuneModel

__all__ = [
    "DatasetSummary",
    "FineTuneModel",
    "LocalEditDataset",
    "discover_dataset",
    "verify_split_groups",
]
