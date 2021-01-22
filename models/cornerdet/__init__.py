from .cornerdet import SepCornerDet

CORNERDET = {
    'SepCornerDet': SepCornerDet,
}


def get_cornerdet(name, **kwargs):
    return CORNERDET[name](**kwargs)
