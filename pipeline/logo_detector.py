from typing import List, Tuple

from ultralytics import YOLO


# Detection Model (YOLOv8 pretrained)
class LogoDetector:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)
        self.model.conf = 0.25  # Confidence threshold

    def detect_logos(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        results = self.model(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes[:, :4]]
