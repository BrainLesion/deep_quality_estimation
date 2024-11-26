import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from monai.networks.nets import DenseNet121
from numpy.typing import NDArray

from deep_quality_estimation.data_handler import DataHandler
from deep_quality_estimation.enums import View

PACKAGE_DIR = Path(__file__).parent


class DQE:

    def __init__(
        self, device: Optional[torch.device] = None, cuda_devices: Optional[str] = "0"
    ):
        """
        Initialize the Deep Quality Estimation model

        Args:
            device (Optional[torch.device], optional): Device to be used. Defaults to None.
            cuda_devices (Optional[str], optional): Visible CUDA devices, e.g. "0", "0,1". Defaults to "0".
        """

        self.device = self._set_device(
            device=device,
            cuda_devices=cuda_devices,
        )

        self.model = self._load_model()

    def _set_device(self, device: Optional[torch.device], cuda_devices: Optional[str]):
        """
        Set the device to be used for the model

        Args:
            device (Optional[torch.device]): Device
            cuda_devices (Optional[str]): Visible CUDA devices, e.g. "0", "0,1"

        Returns:
            torch.device: Device to be used
        """

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        logger.info(f"Using device: {device}")
        return device

    def _load_model(self):
        """
        Load model weights and return initialized model

        Returns:
            monai.networks.nets.DenseNet121: Model
        """

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

        logger.info(f"Model loaded from {checkpoint_path} and initialized")
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
            Tuple[float, Dict[View, float]]: The predicted mean score and a dict with the scores per view
        """

        # load and preprocess data
        data_handler = DataHandler(
            t1c=t1c, t2=t2, t1=t1, flair=flair, segmentation=segmentation
        )
        dataloader = data_handler.get_dataloader()

        # predict ratings
        scores = {}
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                # assuming batch size 1
                logger.debug(f"Predicting for view: {data['view'][0]}")
                inputs = data["inputs"].float().to(self.device)
                outputs = self.model(inputs)
                scores[data["view"][0]] = outputs.cpu().item()
        mean_score = np.mean(list(scores.values()))
        return mean_score, scores
