import numpy as np
from deep_quality_estimation.core.center_of_mass import (
    View,
    compute_center_of_mass,
    get_center_of_mass_slices,
)

from monai.data import Dataset, pad_list_data_collate
from monai.transforms import Compose
from torch.utils.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImageD,
    ToTensord,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    Lambdad,
    ConcatItemsd,
    ConvertToMultiChannelBasedOnBratsClassesd,
)


# TODO make enums for magic strings and numbers
all_channels = ["images", "labels"]
only_images = ["images"]
only_label = ["labels"]

_preprocessing_transforms = Compose(
    [
        # << PREPROCESSING transforms >>
        Lambdad(all_channels, np.nan_to_num),
        ConvertToMultiChannelBasedOnBratsClassesd(
            keys=only_label
        ),  # TODO: adapt this for changed labels?
        ScaleIntensityRangePercentilesd(
            keys=only_images,
            lower=0.5,
            upper=99.5,
            b_min=0,
            b_max=1,
            clip=True,
            relative=False,
            # channel_wise=True,
            channel_wise=False,
        ),
        # Pad all images to 240x240 (coronal and sagittal view will have 240 x155)
        SpatialPadd(
            keys=all_channels, spatial_size=(240, 240), mode="minimum"
        ),  # ensure at least
    ]
)

_postprocessing_transforms = Compose(
    [
        # make tensor
        ConcatItemsd(keys=all_channels, name="inputs", dim=0, allow_missing_keys=False),
        ToTensord(keys=["inputs"]),  # also include target!
    ]
)

inference_transforms = Compose(
    [
        _preprocessing_transforms,
        _postprocessing_transforms,
    ]
)


def get_dataset(t1c, t1, t2, fla, seg):
    center_of_mass = compute_center_of_mass(seg)

    seg_slices = get_center_of_mass_slices(file=seg, center_of_mass=center_of_mass)
    t1c_slices = get_center_of_mass_slices(file=t1c, center_of_mass=center_of_mass)
    t1_slices = get_center_of_mass_slices(file=t1, center_of_mass=center_of_mass)
    t2_slices = get_center_of_mass_slices(file=t2, center_of_mass=center_of_mass)
    fla_slices = get_center_of_mass_slices(file=fla, center_of_mass=center_of_mass)

    data_dicts = []
    for view in View:
        data_dicts.append(
            {
                "t1": t1c_slices[view],
                "t1c": t1_slices[view],
                "t2": t2_slices[view],
                "fla": fla_slices[view],
                "seg": seg_slices[view],
                "images": [
                    t1c_slices[view],
                    t1_slices[view],
                    t2_slices[view],
                    fla_slices[view],
                ],
                "labels": seg_slices[view],
                "view": view.name,
            }
        )

    dataset = Dataset(
        data=data_dicts,
        transform=inference_transforms,
    )
    return dataset


def get_data_loader(t1c, t1, t2, fla, seg):
    dataset = get_dataset(t1c, t1, t2, fla, seg)
    return DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=pad_list_data_collate,
        shuffle=False,
    )
