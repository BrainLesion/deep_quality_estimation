import unittest
from pathlib import Path

import numpy as np

from deep_quality_estimation.data_handler import DataHandler


class TestTransforms(unittest.TestCase):

    def setUp(self):

        self.t1c = Path("tests/data/t1c.nii.gz")
        self.t2 = Path("tests/data/t2w.nii.gz")
        self.t1 = Path("tests/data/t1n.nii.gz")
        self.flair = Path("tests/data/t2f.nii.gz")
        self.segmentation_new_labels = Path("tests/data/seg-BraTS23_1.nii.gz")
        self.segmentation_new_labels_mapped = Path(
            "tests/data/seg-BraTS23_1_mapped.nii.gz"
        )
        self.segmentation_old_labels = Path("tests/data/seg-bratstoolkit_isen.nii.gz")

    def test_labels_transforms_feasible(self):
        """
        Verify that for all segmentations the labels are transformed correctly (have correct shape, one hot and at least 1 labeled pixel for the given data)
        """
        for segmentation in [
            self.segmentation_new_labels_mapped,
            self.segmentation_new_labels,
            self.segmentation_old_labels,
        ]:
            data_handler = DataHandler(
                t1c=self.t1c,
                t2=self.t2,
                t1=self.t1,
                flair=self.flair,
                segmentation=segmentation,
            )

            # get first element from dataloader
            data = next(iter(data_handler.get_dataloader()))

            self.assertEqual(data["labels"].shape, (1, 3, 240, 240))
            for label_map in data["labels"][0]:
                self.assertTrue(np.all(np.isin(label_map, [0, 1])))
                self.assertTrue(np.sum(label_map) > 0)

    def test_labels_transform_equal(self):
        """
        Verify that for the same segmentation (with differing labeling convention) the transformed labels are equal
        """

        data_handler_new_labels = DataHandler(
            t1c=self.t1c,
            t2=self.t2,
            t1=self.t1,
            flair=self.flair,
            segmentation=self.segmentation_new_labels,
        )
        data_handler_new_labels_mapped = DataHandler(
            t1c=self.t1c,
            t2=self.t2,
            t1=self.t1,
            flair=self.flair,
            segmentation=self.segmentation_new_labels_mapped,
        )

        data_new_labels = next(iter(data_handler_new_labels.get_dataloader()))
        data_new_labels_mapped = next(
            iter(data_handler_new_labels_mapped.get_dataloader())
        )

        self.assertTrue(
            np.all(data_new_labels["labels"] == data_new_labels_mapped["labels"])
        )
