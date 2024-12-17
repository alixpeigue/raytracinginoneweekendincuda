import sys

import numpy as np
from PIL import Image


def mse(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    if img1.size != img2.size:
        return

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    mse = np.mean((arr1 - arr2) ** 2)
    print(f"Mean Squared Error: {mse}")
    return mse


mse(sys.argv[1], sys.argv[2])