from pathlib import Path
from typing import List, Union

import nibabel as nib
import numpy as np
import scipy
from numpy.typing import NDArray

from deep_quality_estimation.enums import View


def compute_center_of_mass(
    segmentation: Union[Path, NDArray], fallback_to_edema: bool = True
) -> List[int]:
    """
    Compute the center of mass of the tumor core in the given segmentation.
    If no tumor core is found and the @fallback_to_edema is true, the center of mass of the edema is used as a fallback.

    Args:
        segmentation (Union[Path, NDArray]): Path to the segmentation file or the segmentation data as numpy NDArray.
        fallback_to_edema (bool, optional): Use Edema CoM as fallback if no tumor core is found. Defaults to True.

    Returns:
        List[int]: List of the center of mass coordinates
    """
    if isinstance(segmentation, Path) or isinstance(segmentation, str):
        segmentation = nib.load(segmentation).get_fdata()

    assert isinstance(segmentation, np.ndarray)

    mask = np.zeros(segmentation.shape)

    # get mask of tumor core
    # TODO: verify if this is correct? (label change 4 to 3 in new standard?)
    mask[segmentation == 1] = 1
    mask[segmentation == 3] = 1
    mask[segmentation == 4] = 1

    if (
        np.sum(mask) == 0 and fallback_to_edema
    ):  # if no tumor core is found, use the edema CoM
        mask[segmentation > 0] = 1

    center_of_mass = scipy.ndimage.center_of_mass(mask)
    # convert to int (cuts decimals)
    center_of_mass = [int(x) for x in center_of_mass]
    return center_of_mass


def get_center_of_mass_slices(
    image: Union[Path, NDArray],
    center_of_mass: List[int],
) -> dict[View, NDArray]:
    """
    Get the slices of the given image that contain the center of mass in axial, coronal and sagittal view.

    Args:
        image (Path): Path to the NiFTI image file
        center_of_mass (List[int]): List of the center of mass coordinates

    Returns:
        Dict[View, NDArray]: Dictionary with the views as keys and the 2D image slices as values
    """
    if isinstance(image, Path) or isinstance(image, str):
        image = nib.load(image).get_fdata()

    assert isinstance(image, np.ndarray)

    return {
        View.AXIAL: image[:, :, center_of_mass[2]],
        View.CORONAL: image[:, center_of_mass[1], :],
        View.SAGITTAL: image[center_of_mass[0], :, :],
    }
