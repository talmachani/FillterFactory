#!/usr/bin/python
# coding=utf-8
"""
Author: Talm
Created on 03/05/2018
"""

import numpy as np

from skimage import io, exposure
from skimage import img_as_float
from skimage.filters import gaussian

from filterFactory.core import utils


class Image(object):
    def __init__(self, data, *args, **kwargs) -> None:
        super().__init__()
        self._raw = data
        self._red_channel, self._green_channel, self._blue_channel = self._split_channels(self._raw)

    @classmethod
    def from_file(cls, filepath):
        data = io.imread(filepath)
        data = img_as_float(data)
        return cls(data)

    def _makeme(self, data=None):
        return self.__class__(data=data)

    @staticmethod
    def _split_channels(data):
        _red_channel = data[:, :, 0]
        _green_channel = data[:, :, 1]
        _blue_channel = data[:, :, 2]
        return _red_channel, _green_channel, _blue_channel

    @staticmethod
    def _merge_channels(r, g, b):
        return np.stack([r, g, b], axis=2)

    def show(self, label='Lenna'):
        utils.display(self._raw, label=label)

    # Image Manipulation

    def sharpen(self, a=1.3, b=0.3, sigma=10):
        blurred = gaussian(self._raw, sigma=sigma, multichannel=True)
        sharper = np.clip(self._raw * a - blurred * b, 0, 1.0)
        return self._makeme(sharper)

    def gaussian_blur(self, sigma=10):
        data = gaussian(self._raw, sigma=sigma, multichannel=True)
        return self._makeme(data)

    def contrast_stretching(self, x, y):
        p2, p98 = np.percentile(self._raw, (x, y))
        img_rescale = exposure.rescale_intensity(self._raw, in_range=(p2, p98))
        return self._makeme(data=img_rescale)

    # Equalization
    def equalization(self):
        img_eq = exposure.equalize_hist(self._raw)
        return self._makeme(data=img_eq)

    # Adaptive Equalization
    def adaptive_equalization(self, clip_limit=0.03):
        img_adapteq = exposure.equalize_adapthist(self._raw, clip_limit=clip_limit)
        return self._makeme(img_adapteq)

