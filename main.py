# ===============================
# Food Freshness Detection using CNN
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# -------------------------------
# Device setup (GPU if available)
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# -------------------------------
# Data Transformations
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# -------------------------------
# Load Dataset
# -------------------------------
train_data = datasets.ImageFolder(r'C:\Users\sanji\OneDrive\Desktop\fruits and fresh\dataset\train', transform=transform)
test_data = datasets.ImageFolder(r'C:\Users\sanji\OneDrive\Desktop\fruits and fresh\dataset\test', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print("Classes:", train_data.classes)
print("Training images:", len(train_data))
print("Testing images:", len(test_data))

# -------------------------------
# Define CNN Model
# -------------------------------
class FoodFreshnessCNN(nn.Module):
    def __init__(self):
        super(FoodFreshnessCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, len(train_data.classes))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = FoodFreshnessCNN().to(device)
print("Model created successfully!")

# -------------------------------
# Loss Function and Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# -------------------------------
# Save Model
# -------------------------------
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/food_freshness_model.pth")
print("Model saved successfully!")

# -------------------------------
# Plot Training Loss
# -------------------------------
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig("results/loss_curve.png")
plt.show()


