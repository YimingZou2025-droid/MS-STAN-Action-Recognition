import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_STAN(nn.Module)

    Multi - Scale
    Spatiotemporal
    Attention
    Network(MS - STAN)
    for action recognition.

    def __init__(self, num_classes=10)
        super(MS_STAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128
        56
        56, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    def forward(self, x)
        batch_size = x.shape[0]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))

        x = x.unsqueeze(0)  # Reshape for attention mechanism
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)

        x = self.fc2(x)
        return x


# Example usage
if __name__ == __main__
    model = MS_STAN(num_classes=5)
    sample_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, RGB image
    output = model(sample_input)
    print(fModel
    output
    shape
    {output.shape})
