import os
import cv2
import numpy as np
from numba import jit
from tqdm import tqdm
from osgeo import gdal
from pathlib import Path
from affine import Affine
from shapely.geometry import box
from typing import Union, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


@jit(nopython=True)
def make_windows(
    src_geotransform: List[Union[float, int]],
    src_width: int,
    src_height: int,
    window_size: int = 512,
    stride: float = 0.0,
) -> List[Union[float, int]]:
    """Get sliding window geoinformation for every window in an image (geotransform, image offsets, bounds).

    Args:
        src_geotransform (List[float, float, int, float, int, float]): Source image GDAL GeoTransform
        src_width (int): Source image width in pixels.
        src_height (int): Source image height in pixels.
        window_size (int): Size of sliding window in pixels.
        stride (float): Sliding window overlap, as percentage of window_size, in x and y direction.

    Return:
        geoinfo_list (List[float, float, float, float, int, int, int, int]): List of lists containing window geotransform and bounds information for all image windows
    """
    geotransform = list(src_geotransform)

    geoinfo_list = list()

    for i in range(0, src_width, int(window_size * (1 - stride))):
        if i + window_size > src_width:
            i = src_width - window_size

        for j in range(0, src_height, int(window_size * (1 - stride))):
            if j + window_size > src_height:
                j = src_height - window_size

            ulx = (
                geotransform[1] * i
                + geotransform[2] * j
                + geotransform[1] * 0.5
                + geotransform[2] * 0.5
                + geotransform[0]
            )
            uly = (
                geotransform[4] * i
                + geotransform[5] * j
                + geotransform[4] * 0.5
                + geotransform[5] * 0.5
                + geotransform[3]
            )

            window_geotransform = [
                ulx,
                geotransform[1],
                geotransform[2],
                uly,
                geotransform[4],
                geotransform[5],
            ]

            lrx = ulx + (window_size * window_geotransform[1])
            lry = uly + (window_size * window_geotransform[5])

            geoinfo = [ulx, lry, lrx, uly, i, j, window_size, window_size]

            geoinfo_list.append(geoinfo)

    return geoinfo_list

GeoTransform = Tuple[float, float, float, float, float, float]


def to_shapely_matrix(A: Affine) -> Tuple[float, float, float, float, float, float]:
    """
    Convert an Affine transform into the parameter format expected by
    shapely.affinity.affine_transform.

    Shapely expects parameters in the form:
        [a, b, d, e, xoff, yoff]

    Args:
        A (Affine): Affine transformation matrix.
    Return:
        Tuple[float, float, float, float, float, float]:
            Parameters in Shapely affine format.
    """
    return (A.a, A.b, A.d, A.e, A.c, A.f)


def normalize_geotransform(geotransform: Union[GeoTransform, str]) -> Affine:
    """
    Normalize a GDAL geotransform into an Affine object.

    Args:
        geotransform (Union[GeoTransform, str]):
            GDAL geotransform as a tuple or string representation.

    Return:
        Affine: Affine transformation object.
    """
    if isinstance(geotransform, str):
        geotransform = eval(geotransform)
    return Affine.from_gdal(*geotransform)


def pixel2lonlat(
    geotransform: Tuple[float, float, float, float, float, float], px: float, py: float
) -> Tuple[float, float]:
    """
    Convert Column, Row pixel coordinates to Longitude, Latitude coordinates.

    Args:
        geotransform (Tuple[float]): Image geotransform
        px (float): X pixel coordinate
        py (float): Y pixel coordinate
    Return:
        Tuple[float, float]: X, Y converted to Longitude, Latitude
    """

    if isinstance(geotransform, str):
        affine_transform = Affine.from_gdal(*eval(str(geotransform)))
    else:
        affine_transform = Affine.from_gdal(*geotransform)

    lon, lat = affine_transform * (px, py)

    return lon, lat


def lonlat2pixel(
    geotransform: Tuple[float, float, float, float, float, float], lon: float, lat: float
) -> Tuple[int, int]:
    """
    Convert Longitude, Latitude geocoordinates to Column, Row pixel coordinates.

    Args:
        geotransform (Tuple[float]): Image geotransform
        lon (float): Longitude coordinate
        lat (float): Latitude coordinate
    Return:
        Tuple[int, int]: Longitude, Latitude converted to X, Y pixel coordinates
    """

    if isinstance(geotransform, str):
        affine_transform = Affine.from_gdal(*eval(str(geotransform)))
    else:
        affine_transform = Affine.from_gdal(*geotransform)

    px, py = ~affine_transform * (lon, lat)

    return int(px), int(py)


def chip_image(
    image_path: str, chip_dir: str, window_size: int = 512, stride: float = 0.0
) -> None:
    """
    Chip an image using sliding window with stride.

    Args:
        image_path (str): Path to image
        chip_dir (str): Directory path to write chips to
        window_size (int): Size of sliding window, e.g., 256)
        stride (float): Amount of overlap in x, y direction, e.g., 0.2 for 20% overlap

    Return:
        None
    """
    image_id = os.path.basename(image_path).split(".")[0]
    image = gdal.Open(image_path)
    geotransform = image.GetGeoTransform()
    width = int(image.RasterXSize)
    height = int(image.RasterYSize)

    windows = make_windows(
        geotransform, width, height, window_size=window_size, stride=stride
    )

    for window in windows:
        ulx, lry, lrx, uly, i, j, window_size, window_size = (
            window[0],
            window[1],
            window[2],
            window[3],
            int(window[4]),
            int(window[5]),
            int(window[6]),
            int(window[7]),
        )

        bounds = box(window[0], window[1], window[2], window[3])

        array = image.ReadAsArray(i, j, window_size, window_size)

        window_geotransform = str(
            [
                ulx,
                geotransform[1],
                geotransform[2],
                uly,
                geotransform[4],
                geotransform[5],
            ]
        )

        cent_x, cent_y = array.shape[1] // 2, array.shape[2] // 2

        cent_long, cent_lat = pixel2lonlat(window_geotransform, cent_x, cent_y)
        # print(f"{image_id}_{i}_{j}_{window_size}.png", cent_long, cent_lat)

        # chip_path = os.path.join(chip_dir, f"{image_id}_{i}_{j}_{window_size}_{cent_long}_{cent_lat}.png")
        # tifffile.imwrite(chip_path, arr) # may need planarcongif="CONTIG"

        chip_path = os.path.join(
            chip_dir, f"{image_id}_{i}_{j}_{window_size}_{cent_long}_{cent_lat}.png"
        )
        rgb_array = cv2.cvtColor(
            np.transpose(array, (1, 2, 0)), cv2.COLOR_RGB2BGR
        )  # need to transpose array then rearrange channels
        cv2.imwrite(chip_path, rgb_array)

    return None

def source_images(src: Union[str, Path, List[Union[str, Path]]], valid_exts: List[str] = [".tif", ".nitf", ".ntf"]) -> List[Path]:
    src_list: List[Path] = []

    if isinstance(src, list):
        for p in src:
            path = Path(p)
            if path.is_file() and path.suffix.lower() in valid_exts:
                src_list.append(path)
        return src_list

    path = Path(src)

    if path.is_file():
        if path.suffix.lower() in valid_exts:
            return [path]
        return []

    if path.is_dir():
        return [
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts
        ]

    raise ValueError(f"Invalid src: {src}")

def chip(
    image_dir: str,
    chip_dir: str,
    window_size: int = 512,
    stride: float = 0.0,
    valid_exts: List[str] = [".tif", ".nitf", ".ntf"],
    multiprocess: bool = True,
    overwrite: bool = False,
) -> None:
    """
    Chip a directory of images using sliding window.

    Args:
        image_dir (str): Directory path to images
        chip_dir (str): Directory path to write chips to
        window_size (int): Size of sliding window, e.g., 256
        stride (float): Amount of overlap in x, y direction, e.g., 0.2 for 20% overlap
        valid_exts (List[str]): Image extensions to filter for
        multiprocess (bool): Use multiprocessing vs multithreading

    Return:
        None
    """
    if os.path.exists(chip_dir) and not overwrite:
        return
    
    os.makedirs(chip_dir, exist_ok=True)

    image_paths = source_images(src=image_dir, valid_exts=valid_exts)
    
    if multiprocess:
        with tqdm(
            total=len(image_paths), desc="Chipping images", unit="image"
        ) as progress_bar:
            with ProcessPoolExecutor(max_workers=None) as executor:
                futures = {
                    executor.submit(
                        chip_image, image_path, chip_dir, window_size, stride
                    ): image_path
                    for image_path in image_paths
                }
                for _ in as_completed(futures):
                    progress_bar.update(1)
    else:  # multithread
        with tqdm(
            total=len(image_paths), desc="Chipping images", unit="image"
        ) as progress_bar:
            with ThreadPoolExecutor(max_workers=None) as executor:
                futures = {
                    executor.submit(
                        chip_image, image_path, chip_dir, window_size, stride
                    ): image_path
                    for image_path in image_paths
                }
                for _ in as_completed(futures):
                    progress_bar.update(1)

    return None


if __name__ == "__main__":
    chip()
