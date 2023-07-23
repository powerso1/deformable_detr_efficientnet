# import cv2
# import json
# import base64
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
# from src.status_code import STATUS_CODE
from utils import decode_pil_img, encode_image
import argparse
from main import get_args_parser
import os
from models import build_model
import torch
from inference import init_inference_transform, plot_one_box, make_colors
from models.deformable_detr import PostProcess
import cv2
import random
from draw_comparision import write_text_to_image_comparision

model_dict = {
    "res-50_ddetr": "resnet50",
    "mb-v3L_ddetr": "mobilenet",
    "effi-v2S_ddetr": "efficientnet",
    "swin-T_ddetr": "swin"
}

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

parser = argparse.ArgumentParser(
    'Deformable DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

for k in model_dict.keys():
    backbone = model_dict[k]
    model_path = os.path.join(
        ".", "weight", k, "checkpoint0049.pth")

    # intialize model
    args.backbone = backbone
    model, _, _ = build_model(args)
    model.to(args.device)
    model.eval()
    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])

    model_dict[k] = model

transform = init_inference_transform()
postprocess = PostProcess()
colors = make_colors()


# for k in model_dict.keys():
#     print("*" * 50)
#     print(k, type(model_dict[k]))
#     print(model_dict[k])


app = Flask(__name__)
cors = CORS(app, resources={r"/app/*": {"origins": "*"}})

# load model


def inference(DOC_BYTE, model_name, threshold):
    device = "cuda"
    img_pil = decode_pil_img(DOC_BYTE)
    tensor, _ = transform(image=img_pil, target=None)
    tensor_list = tensor.unsqueeze(0).to(device)

    raw_output = model_dict[model_name](tensor_list)

    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    target_sizes = torch.tensor(
        [[img_cv2.shape[0], img_cv2.shape[1]]]).to(device)
    output = postprocess(raw_output, target_sizes)

    scores = output[0]["scores"].cpu().tolist()
    scores = [round(score, 2) for score in scores]
    labels = output[0]["labels"].cpu().tolist()
    boxes = output[0]["boxes"].int().cpu().numpy().tolist()

    for s, l, b in zip(scores, labels, boxes):
        if s >= threshold:
            plot_one_box(img_cv2, box=b, color=colors[l], label=str(
                CLASSES[l]) + " " + str(s))
    return img_cv2


@app.route("/app/detect", methods=["POST"])
@cross_origin(supports_credentials=True)
def detect():
    try:
        message = "success"
        # debug_mode = True
        return_DOC = []

        request_data = request.get_json(force=True)
        model_name = request_data.get("model_name")
        threshold = request_data.get("threshold")
        DOC_BYTE = request_data.get("document")
        if model_name == "all models":
            result = {k: inference(DOC_BYTE, k, threshold)
                      for k in model_dict.keys()}
            write_text_to_image_comparision(result)
            img1, img2, img3, img4 = result.values()
            left_col = cv2.vconcat([img1, img3])
            right_col = cv2.vconcat([img2, img4])
            result = cv2.hconcat([left_col, right_col])
        else:
            result = inference(DOC_BYTE, model_name, threshold)
        return_DOC = encode_image(result)
    except Exception as e:
        message = "Internal Server Error"
        print(e)
    finally:
        # Generate response data
        response_data = {
            "message": message,
            "result": return_DOC,
        }
        return response_data


if __name__ == "__main__":
    # Start server
    app.run(debug=False, host="0.0.0.0", port=4500)
