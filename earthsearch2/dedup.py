"""Geographic non-maximum suppression for search results.

With overlapping multi-scale chips the same object appears in many chips that
all score well; geographic_nms collapses near-duplicate matches by location so
top_k surfaces distinct objects rather than the same object N ways.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from .chips import ChipSpec


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points, in meters."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def geographic_nms(
    scored_specs: List[Tuple[float, ChipSpec]],
    radius_m: float,
    top_k: int,
) -> List[Tuple[float, ChipSpec]]:
    """Greedy NMS by geographic proximity. Input must be sorted best→worst."""
    if radius_m <= 0:
        return scored_specs[:top_k]
    kept: List[Tuple[float, ChipSpec]] = []
    for score, spec in scored_specs:
        if all(
            haversine_m(spec.lat, spec.lon, k.lat, k.lon) > radius_m
            for _, k in kept
        ):
            kept.append((score, spec))
            if len(kept) >= top_k:
                break
    return kept
