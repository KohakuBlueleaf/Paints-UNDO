import os

import cv2
import numpy as np
from urllib.parse import urlparse


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    return cached_file


def combine_linearts(lineart1: np.ndarray, lineart2: np.ndarray, erode=[False, False]) -> np.ndarray:
    if erode[0]:
        lineart1 = cv2.erode(lineart1, np.ones((3, 3), np.uint8))
    if erode[1]:
        lineart2 = cv2.erode(lineart2, np.ones((3, 3), np.uint8))
    # unify the dark part of lineart1 and lineart2
    union = np.where(lineart1 < lineart2, lineart1, lineart2)
    return union
