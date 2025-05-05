import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Constants
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Custom Dataset for CelebA
class CelebALandmarks(torchvision.datasets.CelebA):
    def __init__(self, root, split="train", transform=None):
        super().__init__(root=root, split=split, target_type="landmarks", download=True, transform=transform)
        self.original_size = (178, 218)

    def __getitem__(self, index):
        image, landmarks = super().__getitem__(index)
        # Normalize landmarks to [0, 1] range
        landmarks = landmarks / torch.tensor([*self.original_size, *self.original_size, *self.original_size, *self.original_size, *self.original_size])
        return image, landmarks[:10]  # Return only 5 points (x, y) pairs

# Load datasets
train_dataset = CelebALandmarks(root="./data", split="train", transform=transform)
val_dataset = CelebALandmarks(root="./data", split="valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Define CNN Model
class LandmarkCNN(nn.Module):
    def __init__(self):
        super(LandmarkCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

# Initialize
model = LandmarkCNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss / len(train_loader):.4f}")

# Evaluation and Visualization
model.eval()
sample_images, sample_labels = next(iter(val_loader))
sample_images = sample_images.to(device)
predictions = model(sample_images).cpu().detach().numpy()
sample_labels = sample_labels.numpy()

# Denormalize landmarks
def denormalize_landmarks(landmarks):
    return (landmarks.reshape(-1, 5, 2) * np.array([178, 218])).astype(int)

def show_predictions(images, true_landmarks, pred_landmarks, count=5):
    plt.figure(figsize=(15, 5))
    for i in range(count):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        true_pts = denormalize_landmarks(true_landmarks[i])
        pred_pts = denormalize_landmarks(pred_landmarks[i])

        img_resized = cv2.resize(img, (178, 218))

        plt.subplot(1, count, i + 1)
        plt.imshow(img_resized)
        for x, y in true_pts:
            plt.plot(x, y, 'go')
        for x, y in pred_pts:
            plt.plot(x, y, 'rx')
        plt.axis('off')
    plt.show()

show_predictions(sample_images, sample_labels, predictions)
