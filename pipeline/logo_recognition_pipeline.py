from typing import List, Tuple

from PIL import Image

from pipeline.logo_classifier import LogoClassifier
from pipeline.logo_detector import LogoDetector


# Inference Pipeline
class LogoRecognitionPipeline:
    def __init__(self, detector: LogoDetector, classifier: LogoClassifier):
        self.detector = detector
        self.classifier = classifier

    def run_on_image(self, image_path: str) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        image = Image.open(image_path).convert("RGB")
        boxes = self.detector.detect_logos(image_path)
        predictions = []

        for box in boxes:
            x1, y1, x2, y2 = box
            cropped = image.crop((x1, y1, x2, y2))
            label, confidence = self.classifier.classify(cropped)
            predictions.append((label, confidence, box))

        return predictions
