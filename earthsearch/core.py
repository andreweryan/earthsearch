import os
import gc
import torch
import faiss
import warnings
import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from typing import Literal, List, Tuple, TypedDict, Union

warnings.filterwarnings("ignore")

gc.enable()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Results(TypedDict):
    path: str
    distance: float


class FeatureExtractor:
    """Extracts embeddings from images using a pre-trained ResNet or DINOv2 model."""

    def __init__(
        self,
        model_type: Literal[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "dinov2_vits14",
            "dinov2_vits14_reg",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ] = "dinov2_vits14_reg",
        device: Literal["cuda", "mps", "cpu"] = "cuda",
    ) -> None:
        """
        Initialize the feature extractor.

        Args:
            model_type: Type of model to use ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "dinov2_vits14", "dinov2_vits14_reg, "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")
            device: Device to use for inference ("cuda": NVIDIA GPU, "mps": Apple M-series chips, "cpu").
        """
        self.device = (
            device
            if device
            else (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.mps.is_available() else "cpu"
            )
        )
        print(f"Using {self.device}")
        self.model_type = model_type

        # Load pre-trained model
        if model_type == "resnet18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.embedding_dim = 512
        elif model_type == "resnet34":
            base_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.embedding_dim = 512
        elif model_type == "resnet50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.embedding_dim = 2048
        elif model_type == "resnet101":
            base_model = models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2
            )
            self.embedding_dim = 2048
        elif model_type == "resnet152":
            base_model = models.resnet152(
                weights=models.ResNet152_Weights.IMAGENET1K_V2
            )
            self.embedding_dim = 2048
        elif model_type == "dinov2_vits14":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", verbose=False
            )
            self.embedding_dim = 384
        elif model_type == "dinov2_vits14_reg":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False
            )
            self.embedding_dim = 384
        elif model_type == "dinov2_vitb14":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", verbose=False
            )
            self.embedding_dim = 768
        elif model_type == "dinov2_vitb14_reg":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14_reg", verbose=False
            )
            self.embedding_dim = 768
        elif model_type == "dinov2_vitl14":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14", verbose=False
            )
            self.embedding_dim = 1024
        elif model_type == "dinov2_vitl14_reg":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitl14_reg", verbose=False
            )
            self.embedding_dim = 1024
        elif model_type == "dinov2_vitg14":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitg14", verbose=False
            )
            self.embedding_dim = 1536
        elif model_type == "dinov2_vitg14_reg":
            base_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitg14_reg", verbose=False
            )
            self.embedding_dim = 1535
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Remove the classification head from ResNet models for feature extraction: don't need class/probabilites
        # and layer just before contains feature rich representation of the image
        if model_type.startswith("resnet"):
            self.model = nn.Sequential(*list(base_model.children())[:-1])
        else:  # if model is a dino model, no classification head to remove so just return the base model
            self.model = base_model

        self.model = self.model.to(self.device)
        self.model.eval()

        # ImageNet Transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def extract_embedding(self, src: Union[str, np.ndarray, PIL.Image]) -> np.array:
        """
        Extract embedding from an image.

        Args:
            src (Union[str, np.ndarray, PIL.Image]): Source image as a file path, numpy array or PIL Image

        Returns:
            features (np.ndarray): Image embedding as a numpy array
        """
        
        if isinstance(src, str):
            image = Image.open(src).convert("RGB")
        elif isinstance(src, np.ndarray):
            image = Image.fromarray(src).convert("RGB")
        else:
            image = src.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        

        with torch.no_grad():
            feats = self.model.forward_features(image_tensor)

        patch_tokens = feats["x_norm_patchtokens"]
        embedding = patch_tokens.mean(dim=1)
        embedding = embedding.squeeze().cpu().numpy()
        embedding /= np.linalg.norm(embedding)

        return embedding


class VectorDatabase:
    """Vector database for storing and searching image embeddings."""

    def __init__(self, embedding_dim: int, index_type: str = "L2") -> None:
        """
        Initialize the vector database.

        Args:
            embedding_dim (int): Dimension of the embedding vectors
            index_type (str): Type of FAISS index to use ("L2")
        """
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        self.image_paths = []
        self.embedding_dim = embedding_dim

    def add_images(self, embeddings: np.ndarray, image_paths: List[str]):
        """
        Add images to the database.

        Args:
            embeddings (np.ndarray): Array of image embeddings
            image_paths (list): Paths to the corresponding images
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.image_paths.extend(image_paths)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> Tuple[float, str]:
        """
        Search for similar images.

        Args:
            query_embedding (np.ndarray): Embedding of the query image
            top_k (int): Number of nearest neighbors to return

        Returns:
            Tuple of (distances, image_paths)
        """
        # Convert to float32 and reshape
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search the index
        distances, indices = self.index.search(
            query_embedding, min(top_k, len(self.image_paths))
        )

        # Get the corresponding image paths
        result_paths = [
            self.image_paths[idx]
            for idx in indices[0]
            if idx >= 0 and idx < len(self.image_paths)
        ]
        result_distances = distances[0].tolist()

        return result_distances, result_paths

    def save_index(self, file_path):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, file_path)

    def load_index(self, file_path):
        """Load the FAISS index from disk."""
        self.index = faiss.read_index(file_path)


class ImageSimilaritySearch:
    """Main class for image similarity search."""

    def __init__(
        self,
        model_type: Literal[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "dinov2_vits14",
            "dinov2_vits14_reg",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ] = "dinov2_vits14_reg",
        index_type: str = "L2",
        device: Literal["cuda", "mps", "cpu"] = "cuda",
    ) -> None:
        """
        Initialize the image similarity search system.

        Args:
            model_type: Type of model to use ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
                "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")
            index_type (str): Type of index to use (currently only using L2)
            device (str): Device to use for inference ("cuda", "mps", "cpu")
        """
        self.index_type = index_type
        self.model_type = model_type
        self.extractor = FeatureExtractor(model_type, device)
        self.db = VectorDatabase(self.extractor.embedding_dim, index_type)

    def index_images(self, image_dir: str, batch_size: int = 32) -> None:
        """
        Index all images in a directory.

        Args:
            image_dir (str): Directory containing images
            batch_size (int): Number of images to process at once
        """

        image_paths = [os.path.join(image_dir, i) for i in os.listdir(image_dir)]

        if not image_paths:
            print(f"No images found in {image_dir}")
            return

        # Process images in batches
        with tqdm(
            total=len(image_paths), desc="Indexing images", unit="image"
        ) as progress_bar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]
                embeddings = []
                valid_paths = []

                for image_path in batch_paths:
                    try:
                        embedding = self.extractor.extract_embedding(image_path)
                        embeddings.append(embedding)
                        valid_paths.append(image_path)
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")

                if embeddings:
                    embeddings_array = np.stack(embeddings)
                    self.db.add_images(embeddings_array, valid_paths)

                # print(f"Indexed {min(i+batch_size, len(image_paths))}/{len(image_paths)} images")
                progress_bar.update(len(batch_paths))

    def find_similar(self, query_image_path: str, top_k: int = 10) -> List[Results]:
        """
        Find images similar to the query image.

        Args:
            query_image_path (str): Path to the query image
            top_k (int): Number of nearest neighbors to return

        Returns:
            List of dictionaries with similarity results
        """
        query_embedding = self.extractor.extract_embedding(query_image_path)
        distances, result_paths = self.db.search(query_embedding, top_k)

        results = []
        for dist, path in zip(distances, result_paths):
            results.append(
                {
                    "path": path,
                    "distance": dist,
                }
            )

        return results

    def save(self, index_path: str, indexed_images_path: str) -> None:
        """Save the vector database index and image paths."""
        # Save the FAISS index
        self.db.save_index(index_path)

        # Save the image paths to file
        with open(indexed_images_path, "w") as f:
            for path in self.db.image_paths:
                f.write(f"{path}\n")

        return indexed_images_path

    def load(self, index_path: str, indexed_images_path: str) -> None:
        """Load the vector database index."""
        self.db.load_index(index_path)

        # Load image paths
        with open(indexed_images_path, "r") as f:
            self.db.image_paths = [line.strip() for line in f.readlines()]


def show_search_results(
    src: Union[str, np.ndarray],
    results: List[dict],
    max_display: int = 10,
    cols: int = 5,
):
    """
    Display search results

    Args:
        src: Path to query image or np.ndarray
        results (List[dict]): Results list containing dicts of nearest neighbor results
        max_display (int): Max number of results to display, typically just top_k
        cols (int): Number of columns in the results grid

    Return:
        None
    """
    if isinstance(src, str):
        query_image = Image.open(src)
    else:
        query_image = src

    num_results = min(max_display, len(results))
    cols = max(1, min(cols, num_results)) if num_results > 0 else 1
    result_rows = int(np.ceil(num_results / cols)) if num_results > 0 else 0

    text_color = "#1a1a1a"
    muted_color = "#6b6b6b"
    border_color = "#d8d8d8"
    bg_color = "#fafafa"

    fig_w = min(2.2 * cols, 13.5)
    fig_h = min(2.0 * result_rows + 3.0, 8.5)
    fig = plt.figure(
        figsize=(fig_w, fig_h),
        dpi=90,
        constrained_layout=True,
        facecolor=bg_color,
    )
    gs = fig.add_gridspec(
        result_rows + 1,
        cols,
        height_ratios=[1.5] + [1.0] * result_rows if result_rows else [1.0],
    )

    def _style_panel(ax, lw=0.8):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(bg_color)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(lw)

    ax_query = fig.add_subplot(gs[0, :])
    ax_query.imshow(query_image)
    ax_query.set_title(
        "QUERY",
        fontsize=10,
        fontweight="bold",
        color=muted_color,
        loc="left",
        pad=10,
    )
    _style_panel(ax_query, lw=1.2)

    for i in range(num_results):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r + 1, c])
        result_path = results[i]["path"]
        distance = results[i]["distance"]
        result_image = Image.open(result_path)
        ax.imshow(result_image)
        ax.set_title(
            f"#{i + 1:02d}",
            fontsize=11,
            fontweight="bold",
            color=text_color,
            loc="left",
            pad=6,
        )
        ax.set_xlabel(
            f"distance  {distance:.3f}",
            fontsize=9,
            color=muted_color,
            labelpad=6,
        )
        _style_panel(ax)

    fig.suptitle(
        "Similarity Search Results",
        fontsize=18,
        fontweight="600",
        color=text_color,
        x=0.02,
        ha="left",
    )
    plt.show()
