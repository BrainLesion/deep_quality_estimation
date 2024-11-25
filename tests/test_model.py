import unittest
from pathlib import Path

import nibabel as nib

from deep_quality_estimation import DQE
from deep_quality_estimation.enums import View


class TestDQEModel(unittest.TestCase):

    def setUp(self):

        self.t1c = Path("tests/data/t1c.nii.gz")
        self.t2 = Path("tests/data/t2w.nii.gz")
        self.t1 = Path("tests/data/t1n.nii.gz")
        self.flair = Path("tests/data/t2f.nii.gz")
        self.segmentation = Path("tests/data/seg-BraTS23_1.nii.gz")

    def test_full_prediction_paths(self):
        dqe = DQE(device="cpu")
        mean_score, scores = dqe.predict(
            t1=self.t1,
            t1c=self.t1c,
            t2=self.t2,
            flair=self.flair,
            segmentation=self.segmentation,
        )
        self.assertAlmostEqual(mean_score, 5.315626939137776, places=4)
        self.assertAlmostEqual(scores[View.AXIAL.name], 5.45253849029541, places=4)
        self.assertAlmostEqual(scores[View.CORONAL.name], 5.1386494636535645, places=4)
        self.assertAlmostEqual(scores[View.SAGITTAL.name], 5.3556928634643555, places=4)

    def test_full_prediction_numpy(self):
        dqe = DQE(device="cpu")
        mean_score, scores = dqe.predict(
            t1=nib.load(self.t1).get_fdata(),
            t1c=nib.load(self.t1c).get_fdata(),
            t2=nib.load(self.t2).get_fdata(),
            flair=nib.load(self.flair).get_fdata(),
            segmentation=nib.load(self.segmentation).get_fdata(),
        )
        self.assertAlmostEqual(mean_score, 5.315626939137776, places=4)
        self.assertAlmostEqual(scores[View.AXIAL.name], 5.45253849029541, places=4)
        self.assertAlmostEqual(scores[View.CORONAL.name], 5.1386494636535645, places=4)
        self.assertAlmostEqual(scores[View.SAGITTAL.name], 5.3556928634643555, places=4)
