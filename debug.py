import datasets.transforms as T
from models.deformable_detr_debug import build
import argparse
import torch
from models.deformable_detr import PostProcess
import matplotlib.pyplot as plt
import cv2
import numpy as np
from main import get_args_parser
import random
from PIL import Image
import sys
import os


model_name_dict = {
    "resnet50": "res-50_ddetr",
    "mobilenet": "mb-v3L_ddetr",
    "efficientnet": "effi-v2S_ddetr",
    "swin": "swin-T_ddetr"
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


def make_colors():
    seed = 2001
    random.seed(seed)
    return [[random.randint(0, 255) for _ in range(3)]
            for _ in range(len(CLASSES))]


def plot_one_box(img, box, color, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 5
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def init_inference_transform():
    """ inspired by "from datasets.coco import make_coco_transforms" """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])


def plot_feauture_map(img, keep, conv_features, dec_attn_weights, bboxes_scaled):
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    # print(dec_attn_weights.shape)
    # print(keep.shape)

    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep, axs.T, bboxes_scaled):
        # ax = ax_i[0]
        # ax.imshow(dec_attn_weights[0, idx].view(h, w))
        # ax.axis('off')
        # ax.set_title(f'query id: {idx.item()}')
        # ax = ax_i[1]
        # ax.imshow(img)
        # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                            fill=False, color='blue', linewidth=3))
        # ax.axis('off')
        # ax.set_title(CLASSES[probas[idx].argmax()])
        pass
    fig.tight_layout()
    fig.savefig('feature_map.png')


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def inference_one_image(model, device, img_path):
    # Read the input image using PIL (RGB) cause transform is designed for PIL
    img = Image.open(img_path).convert("RGB")

    transform = init_inference_transform()
    tensor, _ = transform(image=img, target=None)
    tensor_list = tensor.unsqueeze(0)

    # Convert to CV2 image
    # img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    inputs = tensor_list.to(device)
    outputs = model(inputs)
    # keep only predictions with 0.7+ confidence

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.5

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(
        outputs['pred_boxes'][0, keep].cpu(), img.size)

    # for hook in hooks:
    #     hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # print(conv_features)
    print(enc_attn_weights.shape)
    print(dec_attn_weights.shape)
    print(enc_attn_weights[0][0])

    # print(model.transformer.encoder.layers[-1])
    # print(model.transformer.decoder.layers[-1].cross_attn)

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    colors = COLORS * 100

    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(img)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
    fig.tight_layout()
    fig.savefig('feature_map.png')

    # print(dec_attn_weights)

    # postprocess = PostProcess()
    # target_sizes = torch.tensor(
    #     [[img_cv2.shape[0], img_cv2.shape[1]]]).to(device)
    # output = postprocess(raw_output, target_sizes)

    # scores = output[0]["scores"].cpu().tolist()
    # scores = [round(score, 2) for score in scores]
    # labels = output[0]["labels"].cpu().tolist()
    # boxes = output[0]["boxes"].int().cpu().numpy().tolist()

    # plot_feauture_map(img_cv2, keep, conv_features,
    #                   dec_attn_weights, boxes)

    # colors = make_colors()

    # for s, l, b in zip(scores, labels, boxes):
    #     if s >= 0.2:
    #         plot_one_box(img_cv2, box=b, color=colors[l], label=str(
    #             CLASSES[l]) + " " + str(s))

    return 1


if __name__ == "__main__":
    # set up args parser
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])

    parser.add_argument('--folder', action='store_true',
                        help='Use this flag to process a folder of images')
    parser.add_argument('--inf_path', type=str, default='input.png',
                        help='Path to the input image or folder of images for inference')

    args = parser.parse_args()

    model_path = os.path.join(
        ".", "weight", model_name_dict[args.backbone], "checkpoint0049.pth")

    # intialize model
    model, _, _ = build(args)
    model.to(args.device)

    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt['model'])
    model.eval()

    if args.folder:
        # Get a list of all image files in the folder
        img_files = [os.path.join(args.inf_path, f) for f in os.listdir(
            args.inf_path) if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg')]

        output_folder = os.path.join(
            ".", "output_img", model_name_dict[args.backbone])
        os.makedirs(output_folder, exist_ok=True)

        # Process each image file in the folder
        for img_file in img_files:
            print(img_file)

            # Perform object detection on the image
            result_img = inference_one_image(model, args.device, img_file)

            # Save the image with predicted bounding boxes
            output_path = os.path.join(
                output_folder, os.path.basename(img_file))
            cv2.imwrite(output_path, result_img)
    else:
        # Perform object detection on a single image
        result_img = inference_one_image(model, args.device, args.inf_path)

        # Save the image with predicted bounding boxes
        output_path = "output.png"
        cv2.imwrite(output_path, result_img)
