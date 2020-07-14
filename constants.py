import os
from glob import iglob

DEFAULT_MODEL_PATH = './models/best_model.pth'

MASK_SHAPE = (256, 256)

TIF_EXT = '.tif'

IMAGES_BASE_DIR = './images'
IMAGES_DIR_NAME = 'planet_images'
MASKS_DIR_NAME = 'planet_masks'

IMAGES_PATH = os.path.join(IMAGES_BASE_DIR, IMAGES_DIR_NAME)
MASKS_PATH = IMAGES_PATH.replace(IMAGES_DIR_NAME, MASKS_DIR_NAME)

IMAGE_LOCATION_MAP = {
    os.path.basename(image_loc): (image_loc, mask_loc) for (image_loc, mask_loc) in zip(
        iglob(os.path.join(IMAGES_PATH, f'*{TIF_EXT}')),
        iglob(os.path.join(MASKS_PATH, f'*{TIF_EXT}'))
    )
}
