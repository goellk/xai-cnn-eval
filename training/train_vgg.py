import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Logging
log_file = "training_log_vgg16_imagenet80_0.csv"
with open(log_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Epoch",
        "Train Loss", "Train Accuracy", "Train Precision", "Train Recall", "Train F1",
        "Val Loss", "Val Accuracy", "Val Precision", "Val Recall", "Val F1"
    ])

# Custom VGG16 definition
class VGG16(nn.Module):
    def __init__(self, num_classes=80, weights=None):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(weights=weights)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)
        # Reinitialize classifier head
        init.kaiming_normal_(self.vgg.classifier[6].weight, nonlinearity='relu')
        init.constant_(self.vgg.classifier[6].bias, 0)
        
    def forward(self, x):
        x = self.vgg(x)
        return x

# Define transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)), 
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

transform2 = transforms.Compose([
    transforms.Resize(544),
    transforms.CenterCrop(512),
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


# Load training dataset
workspace_dir = str(os.path.dirname(os.path.dirname(os.getcwd())))
dataset_path = workspace_dir + "/datasets/imagenet80_0/training/" 
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load validation dataset
val_dataset_path = workspace_dir + "/datasets/imagenet80_0/validation/"
val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=transform2)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16(weights=None).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Define scheduler: reduce LR by 0.1 every 30 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_preds = []
    train_labels = []

    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device) 

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # Get the class with highest probability
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total

    # Calculate metrics with average='macro' for multi-class
    train_precision = precision_score(train_labels, train_preds, average='macro')
    train_recall = recall_score(train_labels, train_preds, average='macro')
    train_f1 = f1_score(train_labels, train_preds, average='macro')

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_labels_list = []
    with torch.no_grad():
        for val_images, val_labels in tqdm(val_dataloader):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0)
            val_preds.extend(val_predicted.cpu().numpy())
            val_labels_list.extend(val_labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_dataloader)
    epoch_val_accuracy = val_correct / val_total
    val_precision = precision_score(val_labels_list, val_preds, average='macro')
    val_recall = recall_score(val_labels_list, val_preds, average='macro')
    val_f1 = f1_score(val_labels_list, val_preds, average='macro')

    # Log to CSV
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch + 1,
            epoch_loss, epoch_accuracy, train_precision, train_recall, train_f1,
            epoch_val_loss, epoch_val_accuracy, val_precision, val_recall, val_f1
        ])

    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    print(f"Train     | Loss: {epoch_loss:.4f} | Acc: {epoch_accuracy:.2f} | "
          f"Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | F1: {train_f1:.4f}")
    print(f"Validate  | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_accuracy:.2f} | "
          f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

    # Save model after each epoch
    os.makedirs("models_imagenet80_0", exist_ok=True)
    torch.save(model.state_dict(), f"models_imagenet80_0/vgg16_imagenet80_0_epoch{epoch+1}.pth")
    scheduler.step()
print("Training complete.")
