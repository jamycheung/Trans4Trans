"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .cityscapes import CitySegmentation
from .transparent11 import TransparentSegmentation
from .stanford2d3d import Stanford2d3dSegmentation
from .cocostuff import COCOStuffSegmentation
from .acdc import ACDCSegmentation
from .densepass import DensePASSSegmentation

datasets = {
    'cityscape': CitySegmentation,
    'transparent11': TransparentSegmentation, 
    'stanford2d3d': Stanford2d3dSegmentation,
    'cocostuff': COCOStuffSegmentation,
    'acdc': ACDCSegmentation,
    'densepass': DensePASSSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
