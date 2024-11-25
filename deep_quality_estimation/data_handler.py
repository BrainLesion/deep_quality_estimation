from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from monai.data import Dataset, pad_list_data_collate
from monai.transforms import (
    Compose,
    ConcatItemsd,
    Lambdad,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    ToTensord,
)
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from deep_quality_estimation.center_of_mass import (
    compute_center_of_mass,
    get_center_of_mass_slices,
)
from deep_quality_estimation.enums import View
from deep_quality_estimation.transforms import (
    CustomConvertToMultiChannelBasedOnBratsClassesd,
)


class DataHandler:

    ONLY_IMAGES = ["images"]
    ONLY_LABELS = ["labels"]
    ALL_CHANNELS = [*ONLY_IMAGES, *ONLY_LABELS]

    # ALL_CHANNELS
    def __init__(
        self,
        t1c: Union[Path, NDArray],
        t1: Union[Path, NDArray],
        t2: Union[Path, NDArray],
        flair: Union[Path, NDArray],
        segmentation: Union[Path, NDArray],
    ):
        """
        Initialize the data handler

        Args:
            t1c (Union[Path, NDArray]): Numpy NDArray or Path to the T1c NIfTI file
            t1 (Union[Path, NDArray]): Numpy NDArray or Path to the T1 NIfTI file
            t2 (Union[Path, NDArray]): Numpy NDArray or Path to the T2 NIfTI file
            flair (Union[Path, NDArray]): Numpy NDArray or Path to the FLAIR NIfTI file
            segmentation (Union[Path, NDArray]): Numpy NDArray or Path to the segmentation NIfTI file (In BraTS style)
        """
        self.t1c = t1c
        self.t2 = t2
        self.t1 = t1
        self.flair = flair
        self.segmentation = segmentation

    def _get_transforms(self) -> Compose:
        """
        Returns the transforms to be applied to the dataset

        Returns:
            monai.transforms.Compose: A composition of the transforms
        """
        transforms = Compose(
            [
                Lambdad(self.ALL_CHANNELS, np.nan_to_num),
                CustomConvertToMultiChannelBasedOnBratsClassesd(keys=self.ONLY_LABELS),
                ScaleIntensityRangePercentilesd(
                    keys=self.ONLY_IMAGES,
                    lower=0.5,
                    upper=99.5,
                    b_min=0,
                    b_max=1,
                    clip=True,
                    relative=False,
                    # channel_wise=True,
                    channel_wise=False,
                ),
                # Pad all images to 240x240 (coronal and sagittal view will initially have 240 x155)
                SpatialPadd(
                    keys=self.ALL_CHANNELS, spatial_size=(240, 240), mode="minimum"
                ),  # ensure at least
                # make tensor
                ConcatItemsd(
                    keys=self.ALL_CHANNELS,
                    name="inputs",
                    dim=0,
                    allow_missing_keys=False,
                ),
                ToTensord(keys=["inputs"]),  # also include target!
            ]
        )

        return transforms

    def _compute_slices(
        self,
    ) -> Tuple[
        Dict[View, NDArray],
        Dict[View, NDArray],
        Dict[View, NDArray],
        Dict[View, NDArray],
        Dict[View, NDArray],
    ]:
        """
        Computes the center of mass slices for the different views and all

        Returns:
            Tuple[Dict[View, NDArray], Dict[View, NDArray], Dict[View, NDArray], Dict[View, NDArray], Dict[View, NDArray]]: The slices for the different views
        """

        center_of_mass = compute_center_of_mass(segmentation=self.segmentation)
        t1c_slices = get_center_of_mass_slices(
            image=self.t1c, center_of_mass=center_of_mass
        )
        t1_slices = get_center_of_mass_slices(
            image=self.t1, center_of_mass=center_of_mass
        )
        t2_slices = get_center_of_mass_slices(
            image=self.t2, center_of_mass=center_of_mass
        )
        flair_slices = get_center_of_mass_slices(
            image=self.flair, center_of_mass=center_of_mass
        )
        segmentation_slices = get_center_of_mass_slices(
            image=self.segmentation, center_of_mass=center_of_mass
        )

        return t1c_slices, t1_slices, t2_slices, flair_slices, segmentation_slices

    def _build_dataset(self) -> Dataset:
        """
        Build the dataset consisting of CoM slices for per image for each view.
        i.e. one sample is e.g. a AXIAL view with a 2D slice of each image and segmentation in Brats classes channel format

        Returns:
            Dataset: The dataset
        """

        t1c_slices, t1_slices, t2_slices, flair_slices, segmentation_slices = (
            self._compute_slices()
        )
        data_dicts = []
        for view in View:
            data_dicts.append(
                {
                    "images": [
                        t1c_slices[view],
                        t1_slices[view],
                        t2_slices[view],
                        flair_slices[view],
                    ],
                    "labels": segmentation_slices[view],
                    "view": view.name,
                }
            )

        return Dataset(
            data=data_dicts,
            transform=self._get_transforms(),
        )

    def get_dataloader(self) -> DataLoader:
        """
        Get the dataloader for the dataset

        Returns:
            DataLoader: The dataloader
        """

        dataset = self._build_dataset()

        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=8,
            collate_fn=pad_list_data_collate,
            shuffle=False,
        )
