import os
import sys
import warnings
import argparse
from . import chip, index, search

warnings.simplefilter("once", UserWarning)


def main():
    parser = argparse.ArgumentParser(description="Earth Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    chip_parser = subparsers.add_parser("chip")
    index_parser = subparsers.add_parser("index")
    search_parser = subparsers.add_parser("search")

    chip_parser.add_argument(
        "--image_dir", type=str, required=True, help="Directory path of images to chip"
    )
    chip_parser.add_argument(
        "--chip_dir", type=str, required=True, help="Directory path to write chips to"
    )
    chip_parser.add_argument(
        "--window_size",
        type=int,
        required=False,
        default=512,
        help="Size of sliding window, e.g., 512",
    )
    chip_parser.add_argument(
        "--stride",
        type=float,
        required=False,
        default=0.0,
        help="Amount of overlap in x, y direction, e.g., 0.2",
    )
    chip_parser.add_argument(
        "--valid_exts", nargs="*", required=False, help="Image extensions to filter for"
    )
    chip_parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Use multiprocessing vs multithreading",
    )

    index_parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory path of chipped images to index",
    )
    index_parser.add_argument(
        "--index_path", type=str, required=True, help="Path to save index"
    )
    index_parser.add_argument(
        "--indexed_images_path",
        type=str,
        required=True,
        help="Path to save file containing indexed image paths",
    )
    index_parser.add_argument(
        "--index_type",
        type=str,
        required=True,
        default="L2",
        help="Type of index to use (currently only using L2)",
    )
    index_parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        default="dinov2_vits14_reg",
        help="Type of model to use",
    )
    index_parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device to use for feature extraction (cuda, mps, cpu)",
    )
    index_parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=32,
        help="Number of images per batch",
    )
    index_parser.add_argument(
        "--overwrite_index",
        action="store_true",
        help="Overwrite existing index of same index_path",
    )

    search_parser.add_argument(
        "--query_image",
        type=str,
        required=True,
        help="Path to directory of images to index",
    )
    search_parser.add_argument(
        "--index_path", type=str, required=True, help="Path to saved index"
    )
    search_parser.add_argument(
        "--indexed_images_path",
        type=str,
        required=True,
        help="Path to file containing indexed image paths",
    )
    search_parser.add_argument(
        "--index_type",
        type=str,
        required=True,
        default="L2",
        help="Type of index to use (currently only using L2)",
    )
    search_parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        default=10,
        help="top k-Nearest Neighbors to return",
    )
    search_parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        default="dinov2_vits14_reg",
        help="Type of model to use",
    )
    search_parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device to use for feature extraction (cuda, mps, cpu)",
    )

    chip_parser.set_defaults(func=chip.chip)
    index_parser.set_defaults(func=index.index)
    search_parser.set_defaults(func=search.search)

    args = parser.parse_args()
    if args.command == "chip" and os.path.exists(args.chip_dir):
        warnings.warn("Chip directory exists")
        print(
            "Remove it if you would like a fresh directory."
            "Otherwise, you may have images in your index that you don't not want."
        )

    if args.command:
        func_args = {
            k: v for k, v in vars(args).items() if k not in ("command", "func")
        }
        try:
            args.func(**func_args)
        except NotImplementedError as e:
            print(e)
            sys.exit(-1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
