import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class WeldingDataset(Dataset):
    """
    Custom dataset class for loading welding process data.
    Supports both images and videos.
    """
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: Path to the dataset directory.
        :param transform: Optional transformations (e.g., normalization, augmentation).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data_list = self._load_data_list()

    def _load_data_list(self):
        """ Retrieve all video and image files in the dataset directory. """
        data_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.jpg', '.png')):
                    data_files.append(os.path.join(root, file))
        return data_files

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """ Load a single sample (video frame or image). """
        file_path = self.data_list[idx]
        if file_path.endswith(('.jpg', '.png')):
            img = cv2.imread(file_path)  # Load image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        else:
            cap = cv2.VideoCapture(file_path)  # Load video
            ret, img = cap.read()
            cap.release()
            if not ret:
                raise ValueError(f"Failed to read video file: {file_path}")

        img = cv2.resize(img, (224, 224))  # Resize to a fixed size
        img = img.astype(np.float32) / 255.0  # Normalize pixel values
        img = np.transpose(img, (2, 0, 1))  # Convert to PyTorch format (C, H, W)

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32)

# Example usage
if __name__ == "__main__":
    dataset = WeldingDataset(data_dir="./data/raw/")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
