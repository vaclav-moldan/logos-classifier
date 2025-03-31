# Logo Detection and Classification – Proof of Concept

## Overview

This repository contains a simple proof-of-concept (PoC) system for detecting and classifying logos in images.  
It is based on a two-stage pipeline using:

- **YOLOv8** for logo region detection (trained as a single-class "logo" detector)
- **ResNet18** for logo classification (fine-tuned to distinguish Adidas and Nike)

Currently, the system is trained and tested on **synthetic data** with only two logos.

---

## Example Usage

Run detection and classification on a folder of images:

```bash
python run_logo_pipeline.py
