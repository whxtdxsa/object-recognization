from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_kmnist_dataloaders(batch_size=64):
    transform = transforms.Compose([
        # transforms.Resize((224, 244)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    # train_dataset = Subset(train_dataset, range(64))
    # test_dataset = Subset(test_dataset, range(64))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader
