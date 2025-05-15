import cv2
from utils.preprocess import preprocess_image
from model.yolov5n import YOLOv5nWrapper

img = cv2.imread('sample.jpg')
x = preprocess_image(img)
model = YOLOv5nWrapper('yolov5n.pt')
pred = model.predict(x)


from utils.postprocess import non_max_suppression

raw_pred = model.predict(x)
results = non_max_suppression(raw_pred)

for det in results[0]:
    x1, y1, x2, y2, conf, cls = det
    print(f"class: {int(cls)}, confidence: {conf:.2f}, box: {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")
