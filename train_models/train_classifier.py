import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def train_logo_classifier(data_dir: str, num_classes: int, output_path: str, epochs: int = 10, batch_size: int = 32):
    # Resize and perform color transform on each image
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load image folder
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize pretrained resnet18, replace the final layer
    model = models.resnet18(pretrained=True)

    # Optionally we could freeze the layers so they are not used during training
    # for param in model.parameters():
    #     param.requires_grad = False  # freeze everything

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Define training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


# Example usage
if __name__ == '__main__':
    train_logo_classifier(data_dir=r'/classifier_data', num_classes=2, output_path=r'/models/logo_classifier.pth')
