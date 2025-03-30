from typing import List, Tuple

import torch
from PIL import Image
from torchvision import models as models, transforms as T


# Classification Model (Pretrained ResNet fine-tuned for logo classification)
class LogoClassifier:
    def __init__(self, model_path: str, class_names: List[str]):
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.class_names = class_names

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def classify(self, cropped_image: Image.Image) -> Tuple[str, float]:
        input_tensor = self.transform(cropped_image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            max_prob, pred = torch.max(probs, dim=1)
        return self.class_names[pred.item()], max_prob.item()
