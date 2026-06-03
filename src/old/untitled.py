from __future__ import annotations
import matplotlib.pyplot as plt
import os
import cv2
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
i

def black_to_transparent(image: np.ndarray | str | Path, output_path: str | Path) -> None:
    """Replace true black [0, 0, 0] pixels with transparent and save as PNG."""
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGBA")
    else:
        # Accept BGR (OpenCV) or RGB numpy array
        if image.shape[2] == 3:
            img = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGBA))
        else:
            img = Image.fromarray(image.astype(np.uint8)).convert("RGBA")
    arr = np.array(img)
    black = (arr[:, :, 0] == 0) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)
    arr[black, 3] = 0
    Image.fromarray(arr).save(output_path)
    
