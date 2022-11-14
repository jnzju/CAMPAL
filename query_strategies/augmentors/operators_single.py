import math
import torch
import PIL

from torch import Tensor
from typing import List, Optional

from torchvision.transforms import functional as F, InterpolationMode
import numpy as np


class Cutout(object):
    def __init__(self, n_holes=1, length=0.2):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        if type(img) == torch.Tensor:
            h = img.size(1)
            w = img.size(2)
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                cut_size = int(min(w, h) * self.length)
                y1 = np.clip(y - cut_size // 2, 0, h)
                y2 = np.clip(y + cut_size // 2, 0, h)
                x1 = np.clip(x - cut_size // 2, 0, w)
                x2 = np.clip(x + cut_size // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        else:
            w, h = img.size
            cut_size = int(min(w, h) * self.length)
            x0 = np.random.uniform(0, w)
            y0 = np.random.uniform(0, h)
            x0 = int(max(0, x0 - cut_size / 2.))
            y0 = int(max(0, y0 - cut_size / 2.))
            x1 = int(min(w, x0 + cut_size))
            y1 = int(min(h, y0 + cut_size))
            xy = (x0, y0, x1, y1)
            color = (127, 127, 127)
            img = img.copy()
            PIL.ImageDraw.Draw(img).rectangle(xy, color)

        return img


def apply_op(img: Tensor, op_name: str, magnitude: float,
             interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Identity":
        pass
    elif op_name == "CutOut":
        img = Cutout(length=magnitude)(img)
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img
