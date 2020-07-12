import zlib
import numpy as np


def decode(raw_bytes, dtype):

    decompressed = zlib.decompress(raw_bytes)
    data = np.frombuffer(decompressed, dtype=dtype)

    return data
