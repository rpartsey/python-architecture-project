import numpy as np
import zlib
import cv2
from rasterio.io import MemoryFile

def bytes_to_array(binary_image):

    with MemoryFile(binary_image) as mem_file:
        with mem_file.open() as src:
            array = src.read()

    return array

def decode(raw_bytes, dtype):

    decompressed = zlib.decompress(raw_bytes)
    data = np.frombuffer(decompressed, dtype=dtype)

    return data


def encode(image):
    binary_image = image.flatten().tobytes()
    compressed_image = zlib.compress(binary_image)

    return compressed_image

def overlay(image, label, alpha=0.5):
    overlay_image = np.zeros_like(image)
    overlay_image[:, :, 0] = label * 255

    masked_image = cv2.bitwise_and(image, image, mask=(1 - label))

    masked_image = masked_image + overlay_image
    dst = (alpha * (image) + (1 - alpha) * (masked_image)).astype(np.uint8)

    return dst