from main import get_args_parser
from models import build_model
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args.resume = "./exps/mobilenet_v3_deformable_detr_b2/checkpoint0049.pth"
    args.device = "cuda:1"
    args.backbone = "mobilenet"
    args.num_feature_levels = 1
    print(args)

    model, _, postprocess = build_model(args)
    model.to(args.device)
    model.eval()

    if args.resume is not None:
        ckpt = torch.load(
            args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])

    tensor_list = torch.randn(1, 3, 450, 613)
    inputs = tensor_list.to(args.device)
    x = model(inputs)
    y = postprocess(x)
    print(y)
