import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WeldingDataset
from model import MS_STAN

# Training settings
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001

# Load dataset
train_dataset = WeldingDataset(data_dir="../data/processed/")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss function, and optimizer
model = MS_STAN(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        labels = torch.randint(0, 5, (images.shape[0],))  # Random labels for testing
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save trained model
torch.save(model.state_dict(), "../models/ms_stan.pth")
print("Training complete. Model saved.")
