# import cv2
# import json
# import base64
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
from src.status_code import STATUS_CODE
from src.utils import decode_img, encode_image
from src.my_detect import recognize
import sys
import os
import inspect
current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.append("./yolov7")
from utils.torch_utils import select_device
from utils.general import set_logging
from models.experimental import attempt_load
# Logging setup
# logging.setup()

app = Flask(__name__)
cors = CORS(app, resources={r"/yolov7/*": {"origins": "*"}})

# You can give list of classes to filter by name, Be happy you don't have to put class number. ['train','person' ]
classes_to_filter = None
opt = {
    # "weights": "/content/gdrive/MyDrive/yolov7/runs/train/exp/weights/epoch_024.pt", # Path to weights file default weights are for nano model
    # "yaml": "Trash-5/data.yaml",
    "img-size": 640,  # default image size
    "conf-thres": 0.5,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None
}

set_logging()
device = select_device('cpu')

weights_original = "./yolov7.pt"
model_yolov7_orginal = attempt_load(
    weights_original, map_location=device)

weights_fruit = "./yolov7_fruit.pt"
model_yolov7_fruit = attempt_load(
    weights_fruit, map_location=device)


@app.route("/yolov7/fruit", methods=["POST"])
@cross_origin(supports_credentials=True)
def yolov7_fruit():
    try:
        debug_mode = True
        message = "Success"
        return_DOC = []

        request_data = request.get_json(force=True)
        model_type = request_data.get("model_type")  # original or fruit

        DOC_BYTE = request_data.get("document")
        DOC = decode_img(DOC_BYTE)

        if model_type == "original":
            recognize(image=DOC, model=model_yolov7_orginal, device=device)
        elif model_type == "fruit":
            recognize(image=DOC, model=model_yolov7_fruit, device=device)
        else:
            pass
        # DEBUG
        return_DOC = encode_image(DOC)
    except Exception as e:
        message = "Internal Server Error"
        print(e)
    finally:
        # Generate response data
        response_data = {
            "status_code": STATUS_CODE[message],
            "message": message,
            "result": return_DOC,  # DEBUG
        }
        # logging.info("Response Data: {}".format(str(response_data)))
        return response_data, STATUS_CODE[message]


if __name__ == "__main__":
    # Start server
    app.run(debug=False, host="0.0.0.0", port=4000)
