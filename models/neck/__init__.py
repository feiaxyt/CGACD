# Copyright (c) SenseTime. All Rights Reserved.

from .neck import AdjustLayer


NECKS = {
         'AdjustLayer': AdjustLayer
}


def get_neck(name, **kwargs):
   return NECKS[name](**kwargs)
