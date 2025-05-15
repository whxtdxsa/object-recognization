import cv2
from utils.preprocess import preprocess_image
from model.yolov5n import YOLOv5nWrapper

img = cv2.imread('sample.jpg')
x = preprocess_image(img)
model = YOLOv5nWrapper('yolov5n.pt')
pred = model.predict(x)
