# Copyright (c) SenseTime. All Rights Reserved.

from .resnet import resnet18, resnet50


BACKBONES = {
              'resnet50': resnet50,
            }


def get_backbone(name, **kwargs):
  return BACKBONES[name](**kwargs)
