import os
import random
from pathlib import Path
from typing import Literal

from PIL import Image


def _generate_image_and_label(
    logo_paths: list[Path],
    background_paths: list[Path],
    output_dir: str,
    split: Literal["train", "val"],
    index: int
):
    bg_path = random.choice(background_paths)
    logo_path = random.choice(logo_paths)

    bg = Image.open(bg_path).convert("RGB")
    logo = Image.open(logo_path).convert("RGBA")

    bg_w, bg_h = bg.size
    scale = random.uniform(0.2, 0.5)
    logo_w = int(bg_w * scale)
    logo_h = int(logo.height * (logo_w / logo.width))
    logo = logo.resize((logo_w, logo_h))

    max_x = bg_w - logo_w
    max_y = bg_h - logo_h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    bg.paste(logo, (x, y), logo)

    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    image_path = f"{output_dir}/images/{split}/synth_{index:04d}.jpg"
    bg.save(image_path)

    x_center = (x + logo_w / 2) / bg_w
    y_center = (y + logo_h / 2) / bg_h
    width = logo_w / bg_w
    height = logo_h / bg_h

    label_path = f"{output_dir}/labels/{split}/synth_{index:04d}.txt"
    with open(label_path, 'w') as f:
        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def generate_synthetic_data(
    logo_dir: str,
    background_dir: str,
    output_dir: str,
    num_train: int = 80,
    num_val: int = 20
):
    logo_paths = list(Path(logo_dir).glob("*.png"))
    background_paths = list(Path(background_dir).glob("*.png"))

    if not logo_paths or not background_paths:
        print("No logos or backgrounds found.")
        return

    print(f"Generating {num_train} training and {num_val} validation images...")

    for i in range(num_train):
        _generate_image_and_label(logo_paths, background_paths, output_dir, "train", i)

    for i in range(num_val):
        _generate_image_and_label(logo_paths, background_paths, output_dir, "val", i)

    print("Synthetic data generation complete.")


if __name__ == "__main__":
    generate_synthetic_data(
        logo_dir=r'C:\Users\vacla\PycharmProjects\logosClassificator\train_models\synthetic_data_generation\input_data\logos',
        background_dir=r'C:\Users\vacla\PycharmProjects\logosClassificator\train_models\synthetic_data_generation\input_data\backgrounds\val',
        output_dir=r'C:\Users\vacla\PycharmProjects\logosClassificator\train_models\synthetic_data_generation\test_synthetic_data',
        num_train=100,
        num_val=50
    )
