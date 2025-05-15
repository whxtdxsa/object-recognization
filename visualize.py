# utils/visualize.py
import cv2

def draw_detections(img, detections, class_names=None, color=(0, 255, 0), thickness=2):
    """
    img: 원본 이미지 (BGR, numpy)
    detections: tensor of shape (N, 6) → [x1, y1, x2, y2, conf, cls]
    class_names: optional list of class names
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = map(int, det[:6])
        label = f"{class_names[int(cls_id)] if class_names else int(cls_id)} {det[4]:.2f}"
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img
