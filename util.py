import torch
from torchvision import datasets
from torch.utils.data import DataLoader


def load_data(file_path: str, batch_size: int = 32, num_workers: int = 4) -> DataLoader:
    dataset = datasets.ImageFolder(
        root=file_path,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # faster GPU data transfer
    )

    return data_loader