"""On-the-fly window reading from source GeoTIFFs.

Reads pixel windows directly from rasters via GDAL. No PNG chips are written
to disk — the only persisted artifacts in earthsearch2 are the vector index
and its ChipSpec sidecar.

SourceReader opens a GDAL dataset once and reuses it across many reads from
the same source, which is the common pattern during indexing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from affine import Affine
from osgeo import gdal, gdalconst

gdal.UseExceptions()


GeoTransform = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ImageMeta:
    path: str
    width: int
    height: int
    geotransform: GeoTransform


def read_image_meta(path: str) -> ImageMeta:
    ds = gdal.Open(path)
    if ds is None:
        raise IOError(f"GDAL could not open {path}")
    try:
        return ImageMeta(
            path=path,
            width=ds.RasterXSize,
            height=ds.RasterYSize,
            geotransform=tuple(ds.GetGeoTransform()),
        )
    finally:
        ds = None


def pixel_to_lonlat(geotransform: GeoTransform, px: float, py: float) -> Tuple[float, float]:
    affine = Affine.from_gdal(*geotransform)
    x, y = affine * (px, py)
    return float(x), float(y)


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce a GDAL ReadAsArray output into (H, W, 3) uint8."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=0)
    bands = arr.shape[0]
    if bands >= 3:
        arr = arr[:3]
    elif bands == 2:
        arr = np.concatenate([arr, arr[:1]], axis=0)
    elif bands == 1:
        arr = np.repeat(arr, 3, axis=0)

    arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr / 256).astype(np.uint8)
    # Float / int paths: scale by max if it looks like data is in 0..N, else clip
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    hi = float(np.max(finite))
    if hi > 255:
        arr = (arr / hi) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


class SourceReader:
    """Open a source raster once; read many windows.

    When `target_size` is set, the read is overview-aware: GDAL picks the
    closest pre-built overview level and downsamples to `target_size` in one
    call. On a COG-ified source this skips ~16x of native pixels for large
    windows and is typically the biggest win in indexing throughput. On a
    plain GeoTIFF with no overviews, GDAL falls back to native + resample
    (correct, just no speedup).
    """

    def __init__(self, path: str):
        self.path = path
        self.ds = gdal.Open(path)
        if self.ds is None:
            raise IOError(f"GDAL could not open {path}")

    def read(
        self,
        x: int,
        y: int,
        window: int,
        target_size: Optional[int] = None,
        resample_alg: int = gdalconst.GRIORA_Bilinear,
    ) -> Image.Image:
        out = int(target_size) if target_size else int(window)
        arr = self.ds.ReadAsArray(
            xoff=int(x),
            yoff=int(y),
            xsize=int(window),
            ysize=int(window),
            buf_xsize=out,
            buf_ysize=out,
            resample_alg=resample_alg,
        )
        if arr is None:
            raise IOError(f"Could not read window ({x},{y},{window}) from {self.path}")
        return Image.fromarray(_to_rgb_uint8(arr))

    def close(self) -> None:
        self.ds = None

    def __enter__(self) -> "SourceReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_window(
    path: str,
    x: int,
    y: int,
    window: int,
    target_size: Optional[int] = None,
) -> Image.Image:
    """Convenience: open, read one window, close. Prefer SourceReader for many reads."""
    with SourceReader(path) as r:
        return r.read(x, y, window, target_size=target_size)
