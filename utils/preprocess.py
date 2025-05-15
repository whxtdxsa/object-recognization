import cv2
import numpy as np
import torch

def preprocess_image(img: np.ndarray, img_size: int = 640) -> torch.Tensor:
    """
    img: 원본 BGR 이미지 (OpenCV 형식)
    img_size: 모델 입력 사이즈 (기본 640)
    return: torch.Tensor of shape (1, 3, img_size, img_size), normalized to 0~1
    """
    # Resize & pad (letterbox 방식, 비율 유지)
    h0, w0 = img.shape[:2]  # 원본 이미지 크기
    r = img_size / max(h0, w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR → RGB
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

    # HWC → CHW
    img_chw = img_rgb.transpose((2, 0, 1))

    # Normalize and convert to float32 tensor
    img_tensor = torch.from_numpy(img_chw).float() / 255.0

    # Add batch dimension
    return img_tensor.unsqueeze(0)  # shape: (1, 3, img_size, img_size)
