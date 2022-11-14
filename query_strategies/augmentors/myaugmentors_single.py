import torch
from torch import Tensor
from typing import List, Tuple, Optional, Dict
from torchvision.transforms import functional as F, InterpolationMode
from .operators_single import apply_op


class StrengthGuidedAugmentSingle(torch.nn.Module):

    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31,
                 interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None, ablation_aug=None) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.op_history = []
        self.fixed = False
        self.ablation_aug = ablation_aug

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
            "CutOut": (torch.linspace(0.0, 0.5, num_bins), False)
        }

    def forward(self, img: Tensor) -> Tensor:
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        if self.fixed is False:
            for _ in range(self.num_ops):
                op_meta = self._augmentation_space(self.num_magnitude_bins, F.get_image_size(img))
                if self.ablation_aug is None:
                    op_index = int(torch.randint(len(op_meta), (1,)).item())
                    op_name = list(op_meta.keys())[op_index]
                else:
                    op_name = self.ablation_aug
                magnitudes, signed = op_meta[op_name]
                magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
                if signed and torch.randint(2, (1,)):
                    magnitude *= -1.0
                img = apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                self.op_history.append((op_name, magnitude))
            self.fixed = True
        else:
            for op_name, magnitude in self.op_history:
                img = apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def unfixed(self):
        self.op_history = []
        self.fixed = False

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_ops={num_ops}'
        s += ', magnitude={magnitude}'
        s += ', num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)
