"""High-level index + search orchestration.

index_directory walks a folder of source GeoTIFFs, generates multi-scale
ChipSpecs, reads pixel windows on demand, embeds them in batches, and persists
a FaissStore. Nothing is written to disk except the store directory.

search loads a store, embeds the query (with rotation TTA by default), runs
the FAISS lookup, and applies geographic NMS so the returned top_k are
distinct ground features instead of overlapping views of the same object.
"""

from __future__ import annotations

import os
from typing import List, Tuple

from tqdm import tqdm

from .chips import ChipSpec, make_chip_grid
from .dedup import geographic_nms
from .features import FeatureExtractor
from .io import SourceReader, read_image_meta
from .store import FaissStore


VALID_EXTS = {".tif", ".tiff", ".jp2", ".vrt"}


def _iter_source_images(image_dir: str) -> List[str]:
    paths: List[str] = []
    for name in os.listdir(image_dir):
        if os.path.splitext(name)[1].lower() in VALID_EXTS:
            paths.append(os.path.join(image_dir, name))
    return sorted(paths)


def index_directory(
    image_dir: str,
    store_path: str,
    window_sizes: List[int] = (256, 512, 1024),
    stride_fraction: float = 0.5,
    model_type: str = "dinov2_vits14_reg",
    device: str = None,
    batch_size: int = 32,
    overwrite: bool = False,
) -> FaissStore:
    """Walk image_dir, embed multi-scale chips on the fly, persist a FaissStore.

    Args:
        image_dir: Folder of source rasters (GeoTIFF / JP2 / VRT).
        store_path: Directory to write index.faiss + specs.json into.
        window_sizes: Pixel window sizes to tile at; each becomes a separate
            scale in the index.
        stride_fraction: Step between chip origins as a fraction of window.
            0.5 = 50% overlap (catches boundary objects), 1.0 = no overlap.
        model_type: DINOv2 variant name.
        device: "cuda" | "mps" | "cpu" | None (auto).
        batch_size: Images per forward pass.
        overwrite: If False and store_path exists, load and return without
            re-indexing.
    """
    if os.path.exists(os.path.join(store_path, "index.faiss")) and not overwrite:
        print(f"Store already exists at {store_path}; pass overwrite=True to rebuild.")
        return FaissStore.load(store_path)

    image_paths = _iter_source_images(image_dir)
    if not image_paths:
        raise ValueError(f"No source rasters found in {image_dir}")

    extractor = FeatureExtractor(model_type=model_type, device=device)
    store = FaissStore(embedding_dim=extractor.embedding_dim)

    total_chips = 0
    for src in tqdm(image_paths, desc="Sources"):
        try:
            meta = read_image_meta(src)
        except IOError as e:
            print(f"skip {src}: {e}")
            continue

        specs = list(make_chip_grid(meta, window_sizes, stride_fraction))
        if not specs:
            continue

        with SourceReader(src) as reader:
            batch_specs: List[ChipSpec] = []
            batch_imgs = []
            for spec in specs:
                try:
                    img = reader.read(spec.x, spec.y, spec.window)
                except IOError as e:
                    print(f"  skip spec at ({spec.x},{spec.y},{spec.window}): {e}")
                    continue
                batch_specs.append(spec)
                batch_imgs.append(img)
                if len(batch_imgs) >= batch_size:
                    embs = extractor.embed_batch(batch_imgs)
                    store.add(embs, batch_specs)
                    total_chips += len(batch_specs)
                    batch_specs, batch_imgs = [], []
            if batch_imgs:
                embs = extractor.embed_batch(batch_imgs)
                store.add(embs, batch_specs)
                total_chips += len(batch_specs)

    print(f"Indexed {total_chips} chips across {len(image_paths)} sources.")
    store.save(store_path)
    return store


def search(
    query_image: str,
    store_path: str,
    top_k: int = 10,
    model_type: str = "dinov2_vits14_reg",
    device: str = None,
    tta: bool = True,
    dedup_radius_m: float = 50.0,
    over_retrieve: int = 200,
) -> List[Tuple[float, ChipSpec]]:
    """Embed query, FAISS top-N, then geographic NMS down to top_k.

    Returns list of (cosine_similarity, ChipSpec) in score-descending order.
    """
    store = FaissStore.load(store_path)
    extractor = FeatureExtractor(model_type=model_type, device=device)

    if extractor.embedding_dim != store.embedding_dim:
        raise ValueError(
            f"Model embedding dim ({extractor.embedding_dim}) does not match "
            f"store dim ({store.embedding_dim}). Re-index with the same model."
        )

    query = (
        extractor.extract_query_embedding(query_image)
        if tta
        else extractor.extract_embedding(query_image)
    )
    raw = store.search(query, top_k=over_retrieve)
    return geographic_nms(raw, dedup_radius_m, top_k)
