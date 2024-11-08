from enum import Enum, auto
from pathlib import Path
from typing import List, Union

import nibabel as nib
import numpy as np
import scipy
from numpy.typing import NDArray


class View(Enum):
    AXIAL = auto()
    CORONAL = auto()
    SAGITTAL = auto()


def compute_center_of_mass(
    segmentation: Union[Path, NDArray], fallback_to_edema: bool = True
):
    if isinstance(segmentation, Path):
        segmentation = nib.load(segmentation).get_fdata()

    mask = np.zeros(segmentation.shape)

    # TODO: verify if this is correct?
    mask[segmentation == 1] = 1
    mask[segmentation == 3] = 1
    mask[segmentation == 4] = 1

    if (
        np.sum(mask) == 0 and fallback_to_edema
    ):  # if no tumor core is found, use the edema CoM
        mask[segmentation > 0] = 1

    # TODO: differing import between scipy versions?
    center_of_mass = scipy.ndimage.center_of_mass(
        mask
    )  # get center of mass for tumor core
    # convert to int (cuts decimals)
    center_of_mass = [int(x) for x in center_of_mass]
    return center_of_mass


def get_center_of_mass_slices(
    file: Path,
    center_of_mass: List[int],
):
    data = nib.load(file).get_fdata()

    return {
        View.AXIAL: data[:, :, center_of_mass[2]],
        View.CORONAL: data[:, center_of_mass[1], :],
        View.SAGITTAL: data[center_of_mass[0], :, :],
    }
