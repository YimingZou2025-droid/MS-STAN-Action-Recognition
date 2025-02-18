import torch
from torch.utils.data import DataLoader
from dataset import WeldingDataset
from model import MS_STAN

# Load trained model
model = MS_STAN(num_classes=5)
model.load_state_dict(torch.load("../models/ms_stan.pth"))
model.eval()

# Load test dataset
test_dataset = WeldingDataset(data_dir="../data/processed/")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

correct = 0
total = 0

# Evaluation loop
with torch.no_grad():
    for images in test_loader:
        outputs = model(images)
        labels = torch.randint(0, 5, (images.shape[0],))  # Random labels for testing
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
