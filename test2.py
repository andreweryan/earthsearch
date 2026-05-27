from earthsearch2 import index_directory, search, show_results

# Index — one pass, multi-scale, no PNGs written
store = index_directory(
    image_dir="/Users/aryan/dev/data/xview/images",
    store_path="store_xview",
    window_sizes=(256, 512, 1024),
    stride_fraction=0.5,
    model_type="dinov2_vits14_reg",
    device="cpu",
    batch_size=24,
    overwrite=True,
)

# Search — TTA on, geographic NMS at 50m
results = search(
    query_image="tank.png",
    store_path="store_xview",
    top_k=10,
    tta=True,
    dedup_radius_m=50.0,
)

# Visualize — re-reads windows from source TIFFs at display time
show_results(results, query_image="tank.png")
