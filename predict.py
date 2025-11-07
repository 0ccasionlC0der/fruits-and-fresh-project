# ==========================================
# Food Freshness Detection - Presentation Mode
# ==========================================

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import os

# -------------------------------
# 1. Setup and Configuration
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Same transforms as training
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# Load dataset to get class names
train_data = datasets.ImageFolder(r'C:\Users\sanji\OneDrive\Desktop\fruits and fresh\dataset\train', transform=transform)
classes = train_data.classes
print("Classes:", classes)

# -------------------------------
# 2. Define the same CNN model
# -------------------------------
class FoodFreshnessCNN(nn.Module):
    def __init__(self):
        super(FoodFreshnessCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, len(classes))
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# -------------------------------
# 3. Load the Trained Model
# -------------------------------
model = FoodFreshnessCNN().to(device)
model.load_state_dict(torch.load('results/food_freshness_model.pth', map_location=device))
model.eval()

print("✅ Model loaded successfully!")

# -------------------------------
# 4. Show Saved Accuracy (optional)
# -------------------------------
# You can hardcode your final accuracy value from training here
saved_accuracy = 87.3  # <-- replace with your actual result
print(f"Model Accuracy on Test Data: {saved_accuracy:.2f}%")

# -------------------------------
# 5. Let User Choose an Image
# -------------------------------
img_path = input("\nEnter the full path of the image you want to test:\n> ")

if not os.path.exists(img_path):
    print(" File not found! Please check the path.")
else:
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
        prediction = classes[pred.item()]

    print("\nPredicted Class:", prediction)
