import requests
import cv2
import numpy as np
from PIL import Image
import streamlit as st

from constants import FASTAPI_URL, SEGMENTATION_ENDPOINT, MASK_SHAPE
from utils import bytes_to_array, encode, decode, overlay


st.title('U-Net Burned Areas Segmentation')

st.write(
    '''
    Obtain semantic segmentation maps of the satellite image via U-Net implemented in PyTorch.
    This streamlit example uses a Flask service as backend.
    '''
) # description and instructions

binary_file = st.file_uploader('insert image')  # image upload widget

if st.button('Get segmentation map'):
    if binary_file is None:
        st.write("Insert an image!")  # handle case with no image
    else:
        image = bytes_to_array(binary_file)
        encoded_image = encode(image)

        server_url = f'{FASTAPI_URL}/{SEGMENTATION_ENDPOINT}'
        r = requests.post(
            server_url,
            data=encoded_image,
        )

        encoded_mask = r.content
        mask = decode(encoded_mask, dtype=np.uint8).reshape(MASK_SHAPE)

        b, g, r, nir = image
        rgb = np.array([r, g, b]).transpose((1, 2, 0)) # C x H x W -> H x W x C
        rgb = cv2.normalize(rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        masked = overlay(rgb, mask)

        rgb_image = Image.fromarray(rgb)
        masked_image = Image.fromarray(masked)

        st.image([rgb_image, masked_image], width=300)
