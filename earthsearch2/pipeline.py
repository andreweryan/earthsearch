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

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .chips import ChipSpec, make_chip_grid
from .dedup import geographic_nms
from .features import FeatureExtractor
from .io import SourceReader, read_image_meta
from .store import FaissStore


VALID_EXTS = {".tif", ".tiff", ".jp2", ".vrt"}


class _ChipDataset(Dataset):
    """Reads + transforms one chip per __getitem__ call. Designed to live inside
    DataLoader workers: each worker process lazily opens its own GDAL handles
    (GDAL datasets are not fork-safe to share across processes).
    """

    def __init__(self, specs: List[ChipSpec], target_size: int, transform):
        self.specs = specs
        self.target_size = target_size
        self.transform = transform
        # Per-worker cache; populated lazily after fork/spawn.
        self._readers = {}

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]
        reader = self._readers.get(spec.source)
        if reader is None:
            reader = SourceReader(spec.source)
            self._readers[spec.source] = reader
        img = reader.read(spec.x, spec.y, spec.window, target_size=self.target_size)
        return self.transform(img), idx


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
    num_workers: int = 4,
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
        batch_size: Images per forward pass. On GPU with bfloat16, push this
            as high as memory allows (128 on 24GB, 256+ on A100-class).
        num_workers: DataLoader workers reading and preprocessing chips in
            parallel. 0 = single-process (slow, useful for debugging).
            4–8 is a good range; more can hurt on CPU due to contention.
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

    # Build the full spec list upfront. Source-grouped ordering means each
    # DataLoader worker touches few GDAL handles when shuffle=False.
    all_specs: List[ChipSpec] = []
    for src in image_paths:
        try:
            meta = read_image_meta(src)
        except IOError as e:
            print(f"skip {src}: {e}")
            continue
        all_specs.extend(make_chip_grid(meta, window_sizes, stride_fraction))

    if not all_specs:
        raise ValueError("No chips generated — check window_sizes vs source image dimensions.")

    print(f"Generated {len(all_specs)} chip specs across {len(image_paths)} sources.")

    dataset = _ChipDataset(all_specs, extractor.input_size, extractor.transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(extractor.device == "cuda"),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
    )

    total_chips = 0
    for tensor_batch, idx_batch in tqdm(loader, desc="Embedding", total=len(loader)):
        embs = extractor.embed_tensor_batch(tensor_batch)
        batch_specs = [all_specs[i] for i in idx_batch.tolist()]
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
