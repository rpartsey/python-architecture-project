import numpy as np
import torch
from torchvision.transforms import Compose, Normalize # noqa


class ToNumpyInt32:
    def __call__(self, image):
        return image.astype(np.int32)


class FromNumpy:
    def __call__(self, image):
        return torch.from_numpy(image)


class ToTorchFloat:
    def __call__(self, image):
        return image.float()
