import os
import torch
from .core import ImageSimilaritySearch
from typing import Literal, List, TypedDict


class Results(TypedDict):
    path: str
    distance: float


def search(
    query_image: str,
    index_path: str,
    indexed_images_path: str,
    index_type: str = "L2",
    top_k: int = 10,
    model_type: Literal[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "dinov2_vits14",
        "dinov2_vits14_reg",
        "dinov2_vitb14",
        "dinov2_vitb14_reg",
        "dinov2_vitl14",
        "dinov2_vitl14_reg",
        "dinov2_vitg14",
        "dinov2_vitg14_reg",
    ] = "dinov2_vits14_reg",
    device: Literal["cuda", "mps", "cpu"] = "cuda",
    tta: bool = True,
) -> List[Results]:
    """

    Args:
        query_image (str): Path to query for kNN
        index_path (str): Path to saved index
        indexed_images_path (str): Path to file containing indexed image paths
        index_type (str): Type of index to use (currently only using L2)
        top_k (int): top k-Nearest Neighbors to return
        model_type (str): Type of model to use
        device (str): Device to use for feature extraction ("cuda", "mps", "cpu")

    Return:
        results (List[Results]): A list of dictionaries containing:
            - path (str): File path of the indexed image
            - distance (float): Distance between the query image and the result
    """
    if not (os.path.exists(index_path) and os.path.exists(indexed_images_path)):
        print(
            "No index found, check your index path and image paths file or first index a dataset."
        )
        return None

    if "_reg" in model_type and torch.backends.mps.is_available():
        print(
            "Some PyTorch functionality using DINOv2 models with Registers "
            "not supported by MPS yet, reverting to CPU"
        )
        device = "cpu"
    else:
        device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )

    searcher = ImageSimilaritySearch(
        model_type=model_type, index_type=index_type, device=device
    )  # Initialize the similarity search w/same model during indexing
    searcher.load(
        index_path, indexed_images_path
    )  # Load the previously saved index and image paths file
    results = searcher.find_similar(query_image, top_k=top_k, tta=tta)
    print(f"Querying top {top_k} similar images to: {query_image} (TTA={'on' if tta else 'off'})")
    print("Matches:")
    for idx, result in enumerate(results):
        print(f"{idx + 1}: {result["path"]} - Distance: {result["distance"]}")

    return results


if __name__ == "__main__":
    search()
