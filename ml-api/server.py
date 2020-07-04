from flask import Flask, request

from constants import DEFAULT_MODEL_PATH
from ai.segmentation import load_model, evaluate
from utils import decode, encode
import numpy as np

app = Flask(__name__)

model = load_model(DEFAULT_MODEL_PATH)

@app.route('/segmentation', methods=['POST'])
def get_segmentation_map():
    input_image = decode(request.data, dtype=np.uint16).reshape((4, 256, 256))
    mask = evaluate(model, input_image)
    encoded_mask = encode(mask)

    return encoded_mask

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
