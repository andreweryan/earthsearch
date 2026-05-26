# Earth Search

A library for efficient indexing & similarity search of satellite images. 

Earth Search is built on top of Faiss (Facebook AI Similarity Search) with a number of feature extractors for indexing and querying satellite image embeddings. 

<p align="center">
  <img src="assets/queries.png">
</p>

Task List
- [ ] Extract location from knn image filenames, export to geospatial file formats
- [ ] Integrate other feature extraction methods, e.g., SIFT, hashing, etc., have these implemented locally but requires refactoring 
- [ ] Integrate other faiss index types, IVFFlat, etc. 
- [ ] Chip/index without writing chips to disk (possibly requires some other flat file to sit next to the index)
- [ ] Abstract to support other vector database/search solutions, e.g., pgvector, chromadb, pinecone, etc. 
- [ ] Allow user to extend index with new image chips

## Installation:

From pip:

`pip install earthsearch`

From source:

clone the repo

`pip install . `

## Supported Features:

Images for chipping:
* Any geospatial image format, e.g., TIF, NITF, etc.

Images for indexing:
* Any image format, e.g., TIF, PNG, JPEG, etc.
* If you have a directory of chips, you can index them

Models:
* ResNet-18: `resnet18`
* ResNet-34: `resnet34`
* ResNet-50: `resnet50`
* ResNet-101: `resnet101`
* ResNet-152: `resnet152`
* DINOv2 ViT-S/14 distilled: `dinov2_vits14`
* DINOv2 ViT-S/14 distilled with Registers: `dinov2_vits14_reg`
* DINOv2 ViT-B/14 distilled: `dinov2_vitb14`
* DINOv2 ViT-B/14 distilled with Registers: `dinov2_vitb14_reg`
* DINOv2 ViT-L/14 distilled: `dinov2_vitl14`
* DINOv2 ViT-L/14 distilled with Registers: `dinov2_vitl14_reg`
* DINOv2 ViT-g/14: `dinov2_vitg14`
* DINOv2 ViT-g/14 with Registers: `dinov2_vitg14_reg`

Faiss Indices:
* L2

### Planned Features:
Models/Algorithms:
* SIFT (Scale-Invariant Feature Transform)
* ORB (Oriented FAST and Rotated BRIEF)
* Hashing algorithms

Faiss Indices:
* IndexIVFFlat
* IndexHNSWFlat
* IndexLSH
* IndexBinaryFlat
* IndexBinaryIVF



## Usage:

`chip` takes a directory of satellite imagery scenes and creates a directory of chips for indexing. 
If you already have a directory of chips, you can skip this step and run `index` and `search`. 

Package import usage:
```python
from earthsearch.chip import chip
from earthsearch.index import index
from earthsearch.search import search
from earthsearch.core import show_search_results

image_dir = "directory/path/to/satellite/image/scenes"
chip_dir = "directory/path/to/write/image/chips"
window_size = 512
stride = 0.0
valid_exts = ["tif"]

index_path = "path/to/save/index/to.bin"
indexed_images_path = "path/to/write/indexed/image/paths/to.txt"
index_type = "L2"
model_type = "dinov2_vits14_reg"
device = "cuda" # or "mps", "cpu
batch_size = 32

query_image = "path/to/query/for/top_k/similar/images"
top_k = 10

chip(image_dir, chip_dir, window_size, stride, valid_exts, multiprocess=True)
index(chip_dir, index_path, indexed_images_path, index_type, model_type, device, batch_size, overwrite_index=False)
results = search(query_image, index_path, indexed_images_path, index_type, top_k, model_type, device)

for idx, result in enumerate(results):
    print(f"{idx + 1}: {result["path"]} - Distance: {result["distance"]}")
show_search_results(query_image, results, max_display=top_k)

```

CLI usage:
`earthsearch {chip,index,search} ...`

```
usage: earthsearch chip [-h] --image_dir IMAGE_DIR --chip_dir CHIP_DIR
                        [--window_size WINDOW_SIZE] [--stride STRIDE]
                        [--valid_exts [VALID_EXTS ...]] [--multiprocess]

options:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Directory path to images
  --chip_dir CHIP_DIR   Directory path to write chips to
  --window_size WINDOW_SIZE
                        Size of sliding window, e.g., 512
  --stride STRIDE       Amount of overlap in x, y direction, e.g., 0.2
  --valid_exts [VALID_EXTS ...]
                        Image extensions to filter for
  --multiprocess        Use multiprocessing vs multithreading
```

```
usage: earthsearch index [-h] --image_dir IMAGE_DIR --index_path INDEX_PATH
                         --indexed_images_path INDEXED_IMAGES_PATH --index_type INDEX_TYPE
                         --model_type MODEL_TYPE [--device DEVICE] [--batch_size BATCH_SIZE]
                         [--overwrite_index]

options:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        Path to directory of images to index
  --index_path INDEX_PATH
                        Path to save index
  --indexed_images_path INDEXED_IMAGES_PATH
                        Path to save file containing indexed image paths
  --index_type INDEX_TYPE
                        Type of index to use (currently only using L2)
  --model_type MODEL_TYPE
                        Type of model to use
  --device DEVICE       Device to use for feature extraction (cuda, mps, cpu)
  --batch_size BATCH_SIZE
                        Number of images per batch
  --overwrite_index     Overwrite existing index of same index_path
(.venv)  ✝  ~/repos/earthsearch   main 
```

```
usage: earthsearch search [-h] --query_image QUERY_IMAGE --index_path INDEX_PATH
                          --indexed_images_path INDEXED_IMAGES_PATH --index_type INDEX_TYPE
                          --top_k TOP_K --model_type MODEL_TYPE [--device DEVICE]

options:
  -h, --help            show this help message and exit
  --query_image QUERY_IMAGE
                        Path to directory of images to index
  --index_path INDEX_PATH
                        Path to saved index
  --indexed_images_path INDEXED_IMAGES_PATH
                        Path to file containing indexed image paths
  --index_type INDEX_TYPE
                        Type of index to use (currently only using L2)
  --top_k TOP_K         top k-Nearest Neighbors to return
  --model_type MODEL_TYPE
                        Type of model to use
  --device DEVICE       Device to use for feature extraction (cuda, mps, cpu)
```
