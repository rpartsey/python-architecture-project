
"""This page is for searching and viewing the list of awesome resources"""
import streamlit as st
import st_state_patch  # noqa

import numpy as np
import cv2
import rasterio
from PIL import Image
from sklearn.metrics import jaccard_score
from ai.segmentation import evaluate, load_model
from constants import DEFAULT_MODEL_PATH, IMAGE_LOCATION_MAP
from utils import overlay


def write():
    """Writes content to the app"""

    model = load_model(DEFAULT_MODEL_PATH)

    st.title('U-Net Burned Areas Detection')

    image_name = st.selectbox(
        "PlanetScope image name",
        list(IMAGE_LOCATION_MAP.keys())
    )

    state = st.State()
    if not state:
        state.segmentation_button_clicked = False

    if st.button('Get segmentation map') or state.segmentation_button_clicked:
        state.segmentation_button_clicked = True

        image_loc, mask_loc = IMAGE_LOCATION_MAP[image_name]

        with rasterio.open(image_loc) as image_src:
            image = image_src.read()

        mask_pr = evaluate(model, image)

        b, g, r, nir = image
        rgb = np.array([r, g, b]).transpose((1, 2, 0))  # C x H x W -> H x W x C
        rgb = cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # draw segmentation mask over the image
        masked = overlay(rgb, mask_pr)

        rgb_image = Image.fromarray(rgb)
        masked_image = Image.fromarray(masked)

        st.image([rgb_image, masked_image], width=300)

        if st.button('Compare with ground truth'):
            with rasterio.open(mask_loc) as mask_src:
                mask_gt = mask_src.read()[0]

            image_gt = Image.fromarray(mask_gt * 255)
            image_pr = Image.fromarray(mask_pr * 255)

            st.image([image_gt, image_pr], width=300)
            st.write(f'IoU score: {jaccard_score(mask_gt.flatten(), mask_pr.flatten())}')
            state.segmentation_button_clicked = False
