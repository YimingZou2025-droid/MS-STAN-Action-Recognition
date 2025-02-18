import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

class WeldingDataset(Dataset):
    """
    Custom dataset class for welding process action recognition.
    Supports both image and video data.
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
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            cap = cv2.VideoCapture(file_path)
            ret, img = cap.read()
            cap.release()
            if not ret:
                raise ValueError(f"Failed to read video file: {file_path}")

        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        if self.transform:
            img = self.transform(img)

        return torch.tensor(img, dtype=torch.float32)

# Example usage
if __name__ == "__main__":
    dataset = WeldingDataset(data_dir="../data/raw/")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    sample = next(iter(dataloader))
    print(f"Sample batch shape: {sample.shape}")
