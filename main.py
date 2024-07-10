import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# Ensure proper encoding for output
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Mapping labels to their descriptions
label_map = {
    0: 'Non Demented',
    1: 'Very Mild Demented',
    2: 'Mild Demented',
    3: 'Moderate Demented'
}

# Preprocessing function
def preprocess_function(example):
    image = example['image']
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert grayscale to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image)
    
    return {'image': image, 'label': example['label']}

# Load dataset
full_dataset = load_dataset('Falah/Alzheimer_MRI', split='train')

# Define the splits
train_testvalid = full_dataset.train_test_split(test_size=0.3, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

train_dataset = train_testvalid['train']
val_dataset = test_valid['train']
test_dataset = test_valid['test']

# Apply preprocessing to each split
train_dataset = train_dataset.map(preprocess_function)
val_dataset = val_dataset.map(preprocess_function)
test_dataset = test_dataset.map(preprocess_function)

# Convert to PyTorch datasets
train_dataset.set_format(type='torch', columns=['image', 'label'])
val_dataset.set_format(type='torch', columns=['image', 'label'])
test_dataset.set_format(type='torch', columns=['image', 'label'])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model Selection
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes for Alzheimer's stages

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Acc: {val_acc:.2f}%")

# Testing
model.eval()
test_loss = 0
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

test_loss /= len(test_loader)
test_acc = 100. * test_correct / test_total

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# Save the model
torch.save(model.state_dict(), 'alzheimer_classification_model.pth')

print("Training completed and model saved.")