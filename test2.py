from tqdm import tqdm
from pathlib import Path
from earthsearch2 import index_directory, search, show_results

# Index — one pass, multi-scale, no PNGs written
# store = index_directory(
#     image_dir="/home/andrr/dev/data/xview/images",
#     store_path="store_xview",
#     window_sizes=(256, 512, 1024),
#     stride_fraction=0.5,
#     model_type="dinov2_vits14_reg",
#     device="cuda",
#     batch_size=64,
#     overwrite=False,
# )

"""Run search on single image"""
# query_image="/home/andrr/repos/earthsearch/examples/container-ship.png"
# top_k=25

# """Search — TTA on, geographic NMS at 50m"""
# results = search(
#     query_image=query_image,
#     store_path="store_xview",
#     top_k=top_k,
#     tta=True,
#     dedup_radius_m=50.0,
# )

# """Visualize — re-reads windows from source TIFFs at display time"""
# show_results(results, max_display=top_k, query_image=query_image, show=False, save=True, save_dir="figs")

"""Run search on folder of queries"""
queries = "/home/andrr/repos/earthsearch/examples/"
queries = Path(queries)

query_images = [p for p in queries.iterdir() if p.is_file()]

top_k = 25
with tqdm(range(len(query_images))) as pbar:
    for query in query_images:
        pbar.set_description(f"{str(query)}")
        results = search(
        query_image=str(query),
        store_path="store_xview",
        top_k=top_k,
        tta=True,
        dedup_radius_m=50.0,
    )
        show_results(results, max_display=top_k, query_image=str(query), show=False, save=True, save_dir="figs")

        pbar.update(1)