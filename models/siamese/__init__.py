from .siamese import UPChannelSiamese, DepthwiseSiamese


def get_siamese(name, **kwargs):
   SIAMESE = {
        'UPChannelSiamese': UPChannelSiamese,
        'DepthwiseSiamese': DepthwiseSiamese
       }
   return SIAMESE[name](**kwargs)
