"""
earthsearch2 — successor to earthsearch with three goals:

1. Multi-scale chips: index each source image at several window sizes (different
   ground footprints) so a single query can retrieve hits at the right scale
   without manual zoom selection.
2. No-disk chipping: chips are computed as ChipSpec metadata and pixels are
   read on demand from the source raster. The only persisted artifact is the
   vector index plus its sidecar of specs.
3. Pluggable vector store: a small VectorStore protocol with a FAISS backend
   shipped today, intended to be swapped for LanceDB / Qdrant / pgvector when
   metadata filtering and update semantics start to matter.
"""

import os as _os

# Workaround for OpenMP being statically linked by multiple deps on macOS.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .chips import ChipSpec, make_chip_grid
from .io import (
    ImageMeta,
    SourceReader,
    pixel_to_lonlat,
    read_image_meta,
    read_window,
)
from .features import FeatureExtractor
from .store import FaissStore, VectorStore
from .dedup import geographic_nms, haversine_m
from .pipeline import index_directory, search
from .viz import show_results

__all__ = [
    "ChipSpec",
    "make_chip_grid",
    "ImageMeta",
    "SourceReader",
    "pixel_to_lonlat",
    "read_image_meta",
    "read_window",
    "FeatureExtractor",
    "FaissStore",
    "VectorStore",
    "geographic_nms",
    "haversine_m",
    "index_directory",
    "search",
    "show_results",
]
