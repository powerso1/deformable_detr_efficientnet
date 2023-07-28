from models import build_model
import argparse
from main import get_args_parser
from torchinfo import summary
import torch
"""
parser = argparse.ArgumentParser(
    'Deformable DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


args.num_feature_levels = 4
model, _, _ = build_model(args)
# summary(model, input_size=(1, 3, 450, 613), depth=100)
print(model)
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.swin_transformer import _swin_transformer, Swin_T_Weights
from typing import Dict, List
from torchinfo import summary

class BackboneSwinTransformerFixed(nn.Module):
    def __init__(self, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        model = _swin_transformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[7, 7],
            stochastic_depth_prob=0.2,
            weights=Swin_T_Weights.DEFAULT,
            norm_layer=nn.LayerNorm,
            progress=True)
        embed_dim = 96
        for name, parameter in model.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        # Referenced from https://github.com/HDETR/H-Deformable-DETR/blob/master/models/backbone.py
        if return_interm_layers:
            self.strides = [8, 16, 32]
            self.num_channels = [
                embed_dim * 2,
                embed_dim * 4,
                embed_dim * 8,
            ]
        else:
            self.strides = [32]
            self.num_channels = [embed_dim * 8]
        self.body = IntermediateLayerGetter(model,
                                            return_layers={"features": "0"})

x = BackboneSwinTransformerFixed(train_backbone=False, return_interm_layers=False)