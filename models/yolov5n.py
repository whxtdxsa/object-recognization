import torch

class YOLOv5nWrapper:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = device
        self.model = torch.load(weights_path, map_location=device)['model'].float().fuse().eval()

    def predict(self, image: torch.Tensor):
        """
        image: torch.Tensor of shape (1, 3, H, W), normalized to 0~1
        Returns: model predictions (before NMS)
        """
        with torch.no_grad():
            return self.model(image)
