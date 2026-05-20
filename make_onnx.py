# https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt?download=true
from doclayout_yolo import YOLOv10

model = YOLOv10("model.pt")
model.export(format="onnx", imgsz=1024, dynamic=True)
