# train_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt

# Define the neural network architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # Input channels, output channels, kernel size, stride
        self.bn1 = nn.BatchNorm2d(32)        # BatchNorm after conv1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)        # BatchNorm after conv2
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)      # Adjusted input features based on conv output
        self.bn3 = nn.BatchNorm1d(128)       # BatchNorm after fc1
        self.fc2 = nn.Linear(128, 10)        # 10 output classes for digits 0-9
        # Dropout layer
        self.dropout = nn.Dropout(0.1)       # Reduced dropout rate

    def forward(self, x):
        # Convolutional layers with ReLU, BatchNorm, and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        # Fully connected layers with ReLU and BatchNorm
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        # Output layer
        x = self.fc2(x)  # No activation function here
        return x

# Training settings
batch_size = 64
epochs = 5  # Increased from 5 to 15
learning_rate = 0.001

# Data loaders with transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # Rotate images by up to 10 degrees
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),  # Shear and scale transformations
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    './data', train=True, download=True,
    transform=train_transform
)

test_dataset = datasets.MNIST(
    './data', train=False,
    transform=test_transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size, shuffle=False
)

# Initialize the model, loss function, and optimizer
model = DigitClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Lists to keep track of losses and accuracy
train_losses = []
val_losses = []
val_accuracies = []

# Early stopping variables
best_val_loss = float('inf')
patience = 3
trigger_times = 0

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()            # Zero the gradients
        output = model(data)             # Forward pass
        loss = criterion(output, target) # Calculate loss
        loss.backward()                  # Backward pass
        optimizer.step()                 # Update weights
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    val_accuracies.append(accuracy)
    
    print(f'Epoch {epoch}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        # Save the best model
        torch.save(model.state_dict(), 'digit_classifier.pth')
    else:
        trigger_times += 1
        print(f'Early stopping trigger times: {trigger_times}')
        if trigger_times >= patience:
            print('Early stopping!')
            break
    
    scheduler.step()

# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot validation accuracy
plt.figure(figsize=(10,5))
plt.title("Validation Accuracy")
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()
