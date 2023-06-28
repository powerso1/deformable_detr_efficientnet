from main import get_args_parser
from models import build_model
import argparse
import torch
from models.deformable_detr import PostProcess
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args.resume = "weight/res-50_ddetr/checkpoint0049.pth"
    args.device = "cuda:0"
    args.backbone = "resnet50"
    args.num_feature_levels = 1
    print(args)

    model, _, _ = build_model(args)
    model.to(args.device)
    model.eval()

    if args.resume is not None:
        ckpt = torch.load(
            args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['model'])

    tensor_list = torch.randn(1, 3, 300, 400)

    # Convert the tensor to a numpy array
    img_array = tensor_list.numpy()

    # Transpose the array to match the expected format for an image
    img_array = img_array.transpose(0, 2, 3, 1)

    # Render the image using matplotlib
    plt.imshow(img_array[0])

    # Save the image to a file
    plt.savefig('example.png')

    inputs = tensor_list.to(args.device)
    x = model(inputs)
    print(x.keys())
    postprocess = PostProcess()
    target_sizes = torch.tensor([
    [300, 400]]).to(args.device)
    y=postprocess(x,target_sizes)
    print(y)
