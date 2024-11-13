import torch

from monai.networks.nets import DenseNet201, DenseNet121


class DQE:

    def __init__(self, checkpoint_path, device):
        self.model = DenseNet121(
            spatial_dims=2, in_channels=7, out_channels=1, pretrained=False
        )

        multi_gpu = False
        self.device = device
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )  # change to true once extracted

        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        else:
            if "module." in list(checkpoint["model_state"].keys())[0]:
                checkpoint["model_state"] = {
                    k.replace("module.", ""): v
                    for k, v in checkpoint["model_state"].items()
                }

        self.model = self.model.to(self.device)

        self.model.to(self.device)

        # TODO extract chkpt from state dict
        self.model.load_state_dict(checkpoint["model_state"])

    def predict(self, dataloader):
        results = []

        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                # get the inputs and labels
                inputs = data["inputs"].float().to(self.device)

                # we compute it patch wise instead
                outputs = self.model(inputs)
                print(data["view"], outputs)
                results.append(outputs)
        return results
