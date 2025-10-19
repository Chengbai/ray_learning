import os
import torch
import torchvision
import torchvision.transforms as transforms

from filelock import FileLock
from torch.utils.data import Dataset, DataLoader


# Class labels
classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def get_datasets() -> tuple[Dataset, Dataset]:
    """Create datasets for training & validation, download if necessary."""
    with FileLock(os.path.expanduser("~/data.lock")):
        # Download training data from open datasets
        training_set = torchvision.datasets.FashionMNIST(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )

        # Download test data from open datasets
        validation_setdata = torchvision.datasets.FashionMNIST(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )
        return training_set, validation_setdata


def get_dataloaders(batch_size: int = 4) -> tuple[DataLoader, DataLoader]:
    """Create the dataloader for training & validation."""
    training_set, validation_setdata = get_datasets()

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_setdata, batch_size=batch_size, shuffle=False
    )

    return training_loader, validation_loader
