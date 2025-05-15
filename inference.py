# inference.py
import os
import cv2
import torch

from model.yolov5n import YOLOv5nWrapper
from utils.preprocess import preprocess_image
from utils.postprocess import non_max_suppression
from utils.visualize import draw_detections
from utils.coco import COCO_CLASSES  # 클래스 이름 리스트

def run_inference(input_dir='input', output_dir='output', weights='yolov5n.pt', device='cpu'):
    os.makedirs(output_dir, exist_ok=True)

    # 파일 리스트 불러오기
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # 모델 로드
    model = YOLOv5nWrapper(weights, device)

    for file_name in sorted(image_files):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace('.', '_det.'))

        img = cv2.imread(input_path)
        if img is None:
            print(f"[경고] {input_path} 로드 실패.")
            continue

        x = preprocess_image(img)
        preds = model.predict(x)
        detections = non_max_suppression(preds)[0]

        img_with_boxes = draw_detections(img.copy(), detections, class_names=COCO_CLASSES)
        cv2.imwrite(output_path, img_with_boxes)
        print(f"[완료] {output_path} 저장됨.")

if __name__ == "__main__":
    run_inference()
