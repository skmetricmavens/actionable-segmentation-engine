"""
Data module for the segmentation engine.

Contains data loading, schemas, synthetic generation, and customer ID merging.
"""

from src.data.local_loader import (
    LocalDataLoader,
    LoadResult,
    load_local_data,
    load_events_only,
)

__all__ = [
    "LocalDataLoader",
    "LoadResult",
    "load_local_data",
    "load_events_only",
]
