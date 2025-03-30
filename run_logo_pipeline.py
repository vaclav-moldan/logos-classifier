from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from pipeline.logo_recognition_pipeline import LogoRecognitionPipeline
from pipeline.logo_detector import LogoDetector
from pipeline.logo_classifier import LogoClassifier


def draw_predictions(
    image: Image.Image,
    predictions: List[Tuple[str, float, Tuple[int, int, int, int]]]
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    font.size = 10

    for label, confidence, (x1, y1, x2, y2) in predictions:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 5, y1 + 5), f"{label} ({confidence:.2f})", fill="red", font=font)

    return image


def run_pipeline_on_folder(
    detector_weights: str,
    classifier_weights: str,
    class_names: List[str],
    input_folder: str,
    output_folder: str
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    detector = LogoDetector(detector_weights)
    classifier = LogoClassifier(classifier_weights, class_names)
    pipeline = LogoRecognitionPipeline(detector, classifier)

    image_paths = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))
    print(f"Running on {len(image_paths)} images...")

    for image_path in image_paths:
        predictions = pipeline.run_on_image(str(image_path))

        image = Image.open(image_path).convert("RGB")
        annotated = draw_predictions(image, predictions)

        for label, confidence, box in predictions:
            print(f"{image_path.name}: Detected {label} ({confidence:.2f}) at {box}")

        out_path = output_folder / image_path.name
        annotated.save(out_path)

    print("Done.")


if __name__ == "__main__":
    run_pipeline_on_folder(
        detector_weights="models/logos_detector_weights.pt",
        classifier_weights="models/logo_classifier.pth",
        class_names=["adidas", "nike"],
        input_folder=r"C:\Users\vacla\PycharmProjects\logosClassificator\train_models\synthetic_data_generation\validation_synthetic_data\images\val",
        output_folder=r"C:\Users\vacla\PycharmProjects\logosClassificator\detection_output"
    )
