from torchvision import datasets, transforms

# Apply simple transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load dataset
train_data = datasets.ImageFolder(r'C:\Users\sanji\OneDrive\Desktop\fruits and fresh\dataset\train',transform=transform)

# Display class info
print("Classes found:", train_data.classes)
print("Number of training images:", len(train_data))
