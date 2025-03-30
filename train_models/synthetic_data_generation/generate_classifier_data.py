import os
import random
from pathlib import Path

from PIL import Image


def _paste_logo_on_background(
    logo_path: Path,
    background_path: Path,
    size_range: tuple[float, float] = (0.3, 0.6)
) -> Image.Image:
    bg = Image.open(background_path).convert("RGB")
    logo = Image.open(logo_path).convert("RGBA")

    bg_w, bg_h = bg.size
    scale = random.uniform(*size_range)
    logo_w = int(bg_w * scale)
    logo_h = int(logo.height * (logo_w / logo.width))
    logo = logo.resize((logo_w, logo_h))

    x = random.randint(0, bg_w - logo_w)
    y = random.randint(0, bg_h - logo_h)
    bg.paste(logo, (x, y), logo)

    cropped = bg.crop((x, y, x + logo_w, y + logo_h))
    return cropped


def generate_classifier_data(
    logo_dir: str,
    background_dir: str,
    output_dir: str,
    samples_per_class: int = 50
):
    logo_paths = list(Path(logo_dir).glob("*.png"))
    background_paths = list(Path(background_dir).glob("*.png"))

    if not logo_paths or not background_paths:
        print("No logos or backgrounds found.")
        return

    print(f"Generating {samples_per_class} cropped samples per class...")

    for logo_path in logo_paths:
        class_name = logo_path.stem.lower()  # e.g., 'nike.png' → 'nike'
        class_dir = Path(output_dir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            bg_path = random.choice(background_paths)
            cropped_logo = _paste_logo_on_background(logo_path, bg_path)

            out_path = class_dir / f"{class_name}_{i:04d}.jpg"
            cropped_logo.save(out_path)

    print(f"Generated cropped synthetic classification data into '{output_dir}'.")


if __name__ == "__main__":
    generate_classifier_data(
        logo_dir=r'/logos',
        background_dir=r'/backgrounds',
        output_dir=r'/classifier_data',
        samples_per_class=50
    )
