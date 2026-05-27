from earthsearch.chip import chip
from earthsearch.index import index
from earthsearch.search import search
from earthsearch.core import show_search_results

image_dir = "/home/andrr/dev/data/xview/images"
chip_dir = "/home/andrr/dev/data/chips/xview_512"
window_size = 512
stride = 0.25
valid_exts = [".tif"]

index_path = "dinov2_vitl14_reg-idx.faiss"
indexed_images_path = "dinov2_vitl14_reg-paths.txt"
index_type = "IP"
model_type = "dinov2_vitl14_reg" # dinov2_vits14_reg, dinov2_vitl14_reg, dinov2_vitb14_reg, dinov2_vitg14_reg
device = "cuda" # or "mps", "cpu
batch_size = 64

query_image = "examples/tank.png"
top_k = 25

chip(image_dir, chip_dir, window_size, stride, valid_exts, multiprocess=True, overwrite=False)
index(chip_dir, index_path, indexed_images_path, index_type, model_type, device, batch_size, overwrite_index=False)
results = search(query_image, index_path, indexed_images_path, index_type, top_k, model_type, device)

# for idx, result in enumerate(results):
#     print(f"{idx + 1}: {result["path"]} - Distance: {result["distance"]}")

show_search_results(query_image, results, max_display=top_k)
