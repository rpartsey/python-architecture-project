import torch
import io
from rasterio.io import MemoryFile

from ai.transforms import Compose, ToNumpyInt32, FromNumpy, ToTorchFloat, Normalize


def load_model(model_path, location='cpu'):

    model = torch.load(model_path, map_location=location)

    return model


def evaluate(model, input_image):

    model.eval()

    image_transforms = Compose([
        ToNumpyInt32(),
        FromNumpy(),
        ToTorchFloat(),
        Normalize(
            mean=[4693.149574344914, 4083.8567912125004, 3253.389157030059, 4042.120897153529],
            std=[533.0050173177232, 532.784091756862, 574.671063551312, 913.357907430358]
        )
    ])

    input_image = image_transforms(input_image)
    batch = input_image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(batch)
        predictions = (outputs > 0.5).type(torch.uint8)

    mask = predictions.squeeze().cpu().numpy()

    return mask
