import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import os


class BirdDataset(Dataset):
    def __init__(self, dataframe, img_dir, attributes_tensor, transform=None, is_train=True):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.attributes_tensor = attributes_tensor
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if self.is_train:
            img_name = str(self.dataframe.iloc[idx, 0]).lstrip('/')
            img_path = os.path.join(self.img_dir, img_name)
            label = self.dataframe.iloc[idx, 1] - 1  # Convert to 0-indexed
            attributes = self.attributes_tensor[label]
        else:
            img_name = str(self.dataframe.iloc[idx, 1]).lstrip('/')
            img_path = os.path.join(self.img_dir, img_name)
            label = str(self.dataframe.iloc[idx, 0])
            attributes = torch.zeros(self.attributes_tensor.size(1))
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, attributes

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=200, attribute_dim=312):
        super(BirdClassifier, self).__init__()
        # Use EfficientNet-B4 as backbone
        self.backbone = models.efficientnet_b4(pretrained=True)
        backbone_out = 1792  # EfficientNet-B4 output features
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:  # Keep last few layers trainable
            param.requires_grad = False
            
        self.backbone.classifier = nn.Identity()
        
        # Attribute processing path
        self.attribute_processor = nn.Sequential(
            nn.Linear(attribute_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Combine features
        self.classifier = nn.Sequential(
            nn.Linear(backbone_out + 256, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x, attributes):
        # Process image
        x = self.backbone(x)
        
        # Process attributes
        attr_features = self.attribute_processor(attributes)
        
        # Combine features
        combined = torch.cat((x, attr_features), dim=1)
        output = self.classifier(combined)
        
        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels, attributes) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            attributes = attributes.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, attributes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, '
                      f'Batch {i+1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, ')
            
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, attributes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                attributes = attributes.to(device)
                
                outputs = model(images, attributes)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {running_loss/len(train_loader):.4f}')
        print(f'Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            
    return model

# Data augmentation and normalization
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(380),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(380),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load data and create datasets
attributes = np.load("./data/attributes.npy")
attributes_tensor = torch.tensor(attributes, dtype=torch.float32)

train_data = pd.read_csv("./data/train_images.csv")
test_data = pd.read_csv("./data/test_images_path.csv")

# Split train data into train and validation
train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

train_dataset = BirdDataset(train_dataset, "./data/train_images", attributes_tensor, transform=train_transform)
val_dataset = BirdDataset(val_dataset, "./data/train_images", attributes_tensor, transform=val_transform)
test_dataset = BirdDataset(test_data, "./data/test_images", attributes_tensor, transform=val_transform, is_train=False)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize model and training components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = BirdClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20, device=device)

# Generate predictions
model.eval()
predictions = []

with torch.no_grad():
    for images, ids, attributes in test_loader:
        images = images.to(device)
        attributes = attributes.to(device)
        
        outputs = model(images, attributes)
        _, predicted = outputs.max(1)
        predictions.extend(zip(ids, predicted.cpu().numpy() + 1))

# Create submission file
submission_df = pd.DataFrame(predictions, columns=['id', 'label'])
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully")