# utils/csv_writer.py
import csv
import os

def save_detections_to_csv(csv_path, img_name, detections):
    """
    detections: tensor of shape (N, 6) → [x1, y1, x2, y2, conf, cls]
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['image', 'class', 'confidence', 'x1', 'y1', 'x2', 'y2'])

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            writer.writerow([img_name, int(cls), round(conf, 4), round(x1), round(y1), round(x2), round(y2)])
