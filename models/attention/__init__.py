from .attention import PixelAttention

ATTENTION = {
    'PixelAttention': PixelAttention,
}


def get_attention(name, **kwargs):
    return ATTENTION[name](**kwargs)
