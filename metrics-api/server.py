from flask import Flask, request
import numpy as np
from sklearn.metrics import jaccard_score
from utils import decode


app = Flask(__name__)


@app.route('/iou', methods=['POST'])
def compute_iou():
    mask_gt, mask_pr = decode(request.data, dtype=np.uint8).reshape((2, 256, 256))
    iou = jaccard_score(mask_gt.flatten(), mask_pr.flatten())

    return {'IoU': iou}


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)
