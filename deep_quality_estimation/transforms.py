from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import MapTransform, Transform
from monai.utils.enums import TransformBackends


class CustomConvertToMultiChannelBasedOnBratsClasses(Transform):
    """
    Adapted from Monai's ConvertToMultiChannelBasedOnBratsClasses since label 4 changed to 3 in newer datasets

    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor (used to be label 4 in older datasets)
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # adapted mapping
        result = [
            (img == 1) | (img == 3) | (img == 4),
            (img == 1) | (img == 4) | (img == 3) | (img == 2),
            (img == 4) | (img == 3),
        ]
        # merge labels 1 (tumor non-enh) and 3 (used to be 4) (tumor enh) and 2 (large edema) to WT
        # label 3 (used to be 4) is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )


class CustomConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Adapted from Monai's ConvertToMultiChannelBasedOnBratsClassesd

    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor (used to be label 4 in older datasets)
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = CustomConvertToMultiChannelBasedOnBratsClasses.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = CustomConvertToMultiChannelBasedOnBratsClasses()

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
