"""ChipSpec — the lightweight identifier for a region of a source raster.

A ChipSpec is *not* an image. It records where to find a chip (source path +
pixel offset + window size) and what it represents geographically (lon/lat of
its center). Pixels are read on demand via io.SourceReader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterator, List, Tuple

from .io import ImageMeta, pixel_to_lonlat


@dataclass(frozen=True)
class ChipSpec:
    source: str
    x: int
    y: int
    window: int
    lon: float
    lat: float

    @property
    def center_px(self) -> Tuple[int, int]:
        return self.x + self.window // 2, self.y + self.window // 2

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ChipSpec":
        return cls(
            source=d["source"],
            x=int(d["x"]),
            y=int(d["y"]),
            window=int(d["window"]),
            lon=float(d["lon"]),
            lat=float(d["lat"]),
        )


def make_chip_grid(
    meta: ImageMeta,
    window_sizes: List[int] = (256, 512, 1024),
    stride_fraction: float = 0.5,
) -> Iterator[ChipSpec]:
    """Yield ChipSpecs across an image at multiple window sizes.

    stride_fraction is the step between chip origins as a fraction of the
    window size. 1.0 = no overlap; 0.5 = 50% overlap. Same value applied to
    every scale — call repeatedly with different settings if you need
    per-scale strides.

    Windows are clipped to the image bounds; partial windows at the right/
    bottom edges are dropped.
    """
    for window in window_sizes:
        if window > meta.width or window > meta.height:
            continue
        stride = max(1, int(round(window * stride_fraction)))
        for y in range(0, meta.height - window + 1, stride):
            for x in range(0, meta.width - window + 1, stride):
                cx = x + window // 2
                cy = y + window // 2
                lon, lat = pixel_to_lonlat(meta.geotransform, cx, cy)
                yield ChipSpec(
                    source=meta.path,
                    x=x,
                    y=y,
                    window=window,
                    lon=lon,
                    lat=lat,
                )
