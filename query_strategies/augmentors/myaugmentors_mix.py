import torch
from torch import Tensor
from typing import Tuple, Dict, List
from torchvision.transforms import functional as F
from .operators_mix import apply_op


class StrengthGuidedAugmentMixing(torch.nn.Module):

    def __init__(self, magnitude: int = 9, num_magnitude_bins: int = 26, ablation_aug=None) -> None:
        super().__init__()
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.op_history = []
        self.fixed = False
        self.ablation_aug = ablation_aug

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "MixUp": (torch.linspace(0.0, 0.5, num_bins), False),
            "CutMix": (torch.linspace(0.0, 0.5, num_bins), False)
        }

    def forward(self, img_1: Tensor, img_2: Tensor, label_1: Tensor, label_2: Tensor, num_class)\
            -> Tuple[Tensor, Tensor]:

        if self.fixed is False:
            op_meta = self._augmentation_space(self.num_magnitude_bins, F.get_image_size(img_1))
            if self.ablation_aug is None:
                op_index = int(torch.randint(len(op_meta), (1,)).item())
                op_name = list(op_meta.keys())[op_index]
            else:
                op_name = self.ablation_aug
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = apply_op(img_1, img_2, label_1, label_2, num_class, op_name, magnitude)
            self.op_history.append((op_name, magnitude))
            self.fixed = True
        else:
            for op_name, magnitude in self.op_history:
                img = apply_op(img_1, img_2, label_1, label_2, num_class, op_name, magnitude)

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
