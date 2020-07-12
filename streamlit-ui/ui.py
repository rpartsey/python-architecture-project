import requests
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import st_state_patch # noqa

from constants import ML_API_URL, SEGMENTATION_ENDPOINT, MASK_SHAPE, IOU_ENDPOINT, METRICS_API_URL
from utils import bytes_to_array, encode, decode, overlay


st.title('U-Net Burned Areas Segmentation')

st.write(
    '''
    Obtain semantic segmentation maps of the satellite image via U-Net implemented in PyTorch.
    This streamlit example uses a Flask service as backend.
    '''
)  # description and instructions

binary_image = st.file_uploader('Insert image')  # image upload widget

state = st.State()
if not state:
    state.segmentation_button_clicked = False

if st.button('Get segmentation map') or state.segmentation_button_clicked:
    state.segmentation_button_clicked = True

    if binary_image is None:
        st.write("Insert an image!")  # handle case with no image
    else:
        image = bytes_to_array(binary_image)
        encoded_image = encode(image)

        ml_api_server_url = f'{ML_API_URL}/{SEGMENTATION_ENDPOINT}'
        r = requests.post(
            ml_api_server_url,
            data=encoded_image,
        )

        encoded_mask = r.content
        mask_pr = decode(encoded_mask, dtype=np.uint8).reshape(MASK_SHAPE)

        b, g, r, nir = image
        rgb = np.array([r, g, b]).transpose((1, 2, 0)) # C x H x W -> H x W x C
        rgb = cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        masked = overlay(rgb, mask_pr)

        rgb_image = Image.fromarray(rgb)
        masked_image = Image.fromarray(masked)

        st.image([rgb_image, masked_image], width=300)

        binary_mask = st.file_uploader('Insert ground truth mask')

        if st.button('Compare with ground truth'):
            if binary_mask is None:
                st.write("Insert an image!")  # handle case with no image
            else:
                mask_gt = bytes_to_array(binary_mask)[0]
                encoded_image = encode(np.stack([mask_gt, mask_pr]))

                image_gt = Image.fromarray(mask_gt * 255)
                image_pr = Image.fromarray(mask_pr * 255)

                st.image([image_gt, image_pr], width=300)

                metrics_api_server_url = f'{METRICS_API_URL}/{IOU_ENDPOINT}'
                r = requests.post(
                    metrics_api_server_url,
                    data=encoded_image,
                )
                st.write(f'IoU score: {r.json()["IoU"]}')
