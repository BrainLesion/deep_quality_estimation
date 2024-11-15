from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import numpy as np
from numpy.typing import NDArray

from monai.networks.nets import DenseNet121

from deep_quality_estimation.enums import View
from deep_quality_estimation.dataloader import DataHandler

import os

PACKAGE_DIR = Path(__file__).parent


class DQE:

    def __init__(
        self, device: Optional[torch.device] = None, cuda_devices: Optional[str] = "0"
    ):

        self.device = torch.device(device)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        self.model = self._load_model()

    # def _set_device(self):

    def _load_model(self):
        checkpoint_path = PACKAGE_DIR / "weights/dqe_weights.pth"
        model = DenseNet121(
            spatial_dims=2, in_channels=7, out_channels=1, pretrained=False
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if self.device == torch.device("cpu"):
            if "module." in list(checkpoint.keys())[0]:
                checkpoint = {
                    k.replace("module.", ""): v for k, v in checkpoint.items()
                }
        else:
            model = torch.nn.parallel.DataParallel(model)

        model.load_state_dict(checkpoint)

        model = model.to(self.device)
        return model

    def predict(
        self,
        t1c: Union[Path, NDArray],
        t1: Union[Path, NDArray],
        t2: Union[Path, NDArray],
        flair: Union[Path, NDArray],
        segmentation: Union[Path, NDArray],
    ) -> Tuple[float, Dict[View, float]]:
        """
        Predict the quality of the given Segmentation

        Args:
            t1c (Union[Path, NDArray]): Numpy NDArray or Path to the T1c NIfTI file
            t1 (Union[Path, NDArray]): Numpy NDArray or Path to the T1 NIfTI file
            t2 (Union[Path, NDArray]): Numpy NDArray or Path to the T2 NIfTI file
            flair (Union[Path, NDArray]): Numpy NDArray or Path to the FLAIR NIfTI file
            segmentation (Union[Path, NDArray]): Numpy NDArray or Path to the segmentation NIfTI file (In BraTS style)

        Returns:
            Tuple[float, Dict[View, float]]: The predicted mean quality and a dict with the predictions per view
        """

        # load and preprocess data
        data_handler = DataHandler(
            t1c=t1c, t2=t2, t1=t1, flair=flair, segmentation=segmentation
        )
        # from deep_quality_estimation.core.func_dataloader import get_data_loader

        dataloader = data_handler.get_dataloader()
        # dataloader = get_data_loader(t1c, t1, t2, flair, segmentation)

        # predict ratings
        scores = {}
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                # assuming batch size 1
                # get the inputs and labels
                inputs = data["inputs"].float().to(self.device)
                outputs = self.model(inputs)
                scores[data["view"][0]] = outputs.cpu().item()
        mean_score = np.mean(list(scores.values()))
        return mean_score, scores
