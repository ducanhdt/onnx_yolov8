# ONNX YOLOv8 Object Detection

This code can run YOLOv8 for object detection in an ONNX environment without requiring the installation of Ultralytics or Torch.

## Requirements
 * Check the **requirements.txt** file.
 * For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu**, otherwise use the **onnxruntime** library.

## Installation
```shell
pip install -r requirements.txt
```

You can convert the model using the following code after installing ultralitics (`pip install ultralytics`):
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt") 
model.export(format="onnx", imgsz=[480,640])
```

## Original YOLOv8 model
The original YOLOv8 model can be found in this repository: [YOLOv8 Repository](https://github.com/ultralytics/ultralytics)
- The License of the models is GPL-3.0 license: [License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

## Examples

 ```shell
 python image_object_detection.py
 ```

## References:
* YOLOv8 model: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* Other Repo: [https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/)