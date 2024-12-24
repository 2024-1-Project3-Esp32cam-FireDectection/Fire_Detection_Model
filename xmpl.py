import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score
from IPython.core.display import Video

model = YOLO(f"\runs\detect\train\weights\best.pt")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed!")

img_counter = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    annotated_image = bounding_box_annotator.annotate(
    scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(
    scene=annotated_image,detections=detections)

    cv2.imshow('Webcam', annotated_image)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break