import zlib
import numpy as np

def decode(raw_bytes, dtype):

    decompressed = zlib.decompress(raw_bytes)
    data = np.frombuffer(decompressed, dtype=dtype)

    return data


def encode(image):
    binary_image = image.flatten().tobytes()
    compressed_image = zlib.compress(binary_image)

    return compressed_image