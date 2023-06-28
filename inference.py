from main import get_args_parser
from models import build_model
import argparse
import torch
from models.deformable_detr import PostProcess
import matplotlib.pyplot as plt
import cv2
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args.backbone = "resnet50"
    args.resume = "weight/res-50_ddetr/checkpoint0049.pth"
    args.device = "cuda:0"
    args.num_feature_levels = 1
    print(args)

    model, _, _ = build_model(args)
    model.to(args.device)
    model.eval()

    if args.resume is not None:
        ckpt = torch.load(
            args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])

    # Read the input image using OpenCV
    img = cv2.imread("input.png")

    # Convert the image to a tensor
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    tensor_list = torch.from_numpy(np.expand_dims(img.transpose(2, 0, 1), axis=0)).float().div(255)

    inputs = tensor_list.to(args.device)
    x = model(inputs)
    postprocess = PostProcess()
    target_sizes = torch.tensor([[img.shape[0], img.shape[1]]]).to(args.device)
    y = postprocess(x, target_sizes)
    # print(y[0]["scores"])
    # print(y[0]["labels"])
    # print(y[0]["boxes"])

    for s,l,b in zip(y[0]["scores"], y[0]["labels"], y[0]["boxes"]):
        print(s,l,b)
        break

    # Draw bounding boxes on image
    for box in boxes:
        x_min, y_min, x_max, y_max = box.int().cpu().numpy()
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


    # Save image with bounding boxes as output.png
    # Convert the RGB image to BGR format
    output_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output.png", output_img)