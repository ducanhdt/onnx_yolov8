import cv2
from yolov8 import YOLOv8

# Initialize yolov8 object detector
model_path = "models/v8n_best.onnx"
img_url = "Capture.PNG"
img = cv2.imread(img_url)

# Detect with onnx model
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.7)
boxes, scores, class_ids = yolov8_detector(img)
print(boxes, scores, class_ids)
x1, y1, x2, y2 = boxes[0]
output_img = cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color=[255,0,0])

try:
    from ultralytics import YOLO
    yolov8_detector_ultra = YOLO(model_path,task='detect',verbose=True)
    result = yolov8_detector_ultra(img, conf=0.5, verbose=True)
    print(result[0].boxes.xyxy.tolist())
    # # Draw detections
    x1, y1, x2, y2 = result[0].boxes.xyxy.tolist()[0]
    output_img = cv2.rectangle(output_img,(int(x1),int(y1)),(int(x2),int(y2)),color=[0,255,0])
except:
    print("Ultralytics isn't installed yet")

cv2.imshow("Visualization", output_img)
cv2.waitKey(0)

