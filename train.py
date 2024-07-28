import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
import argparse
import torch.optim as optim
from torch.optim import lr_scheduler



# Argument parsing
parser = argparse.ArgumentParser(description='Train a ResNet50 model on a specified dataset.')
parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset directory')
args = parser.parse_args()

seed = 114514
torch.manual_seed(seed)

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# Using the dataset path from the argument
train_dataset = datasets.ImageFolder(root=args.dataset, transform=transform)
train_dataset.class_to_idx
total_size = len(train_dataset)
val_size = int(total_size * 0.2)
train_size = total_size - val_size

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=768, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=768, shuffle=False)
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 5)

gpu_ids = [1, 2, 3, 4]

if torch.cuda.device_count() > 1:
    # print("Using", torch.cuda.device_count(), "GPUs!")
    resnet50 = nn.DataParallel(resnet50, device_ids=gpu_ids)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)
best_val_accuracy = 0.0
best_model_state = None

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_epochs = 20
for epoch in range(num_epochs):
    resnet50.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    train_loader_with_progress = tqdm(train_loader)    

    for i, data in enumerate(train_loader_with_progress, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = resnet50(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 30 == 29:    # Print every 50 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}, accuracy: {100 * correct / total:.2f}%')
            running_loss = 0.0
            correct = 0
            total = 0

        train_loader_with_progress.set_description(f'Epoch {epoch+1}/{num_epochs}')

    scheduler.step()  # Update learning rate

    # Evaluate on the validation set at the end of each epoch
    resnet50.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    val_loader_with_progress = tqdm(val_loader)

    with torch.no_grad():
        for data in val_loader_with_progress:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = resnet50(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_loader_with_progress.set_description('Validation')

    val_accuracy = 100 * correct / total

    print(f'Validation loss: {val_loss / len(val_loader):.3f}, Validation accuracy: {val_accuracy:.2f}%')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = resnet50.state_dict()

print('Finished Training')
if best_model_state is not None:
    torch.save(best_model_state, './5class_best_resnet50_model.pth')
    print(best_val_accuracy)
torch.cuda.empty_cache()
y_pred = []
y_true = []

misclassified_images=[]

resnet50.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = resnet50(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.view(-1).tolist())
        y_true.extend(labels.view(-1).tolist())

        # misclassified_images.extend(inputs[predicted != labels].cpu().numpy())


conf_matrix = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')
ax.set_title('Confusion Matrix')
plt.savefig('./5class_resnet50_confusion_matrix.png')
