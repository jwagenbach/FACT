import sys

sys.path.append('./')

import hydra
import torch
from torch import nn
from omegaconf import DictConfig
from pathlib import Path
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

from lfxai.models.images import SimCLR


def evaluate(args: DictConfig):

    # Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = Path.cwd() / f"models/simclr_{args.backbone}_epoch{args.epochs}.pt"
    model = SimCLR(resnet18, projection_dim=args.projection_dim).to(device).eval()
    model.load_state_dict(torch.load(model_path), strict=True)

    # Classifier layer
    classifier = nn.Linear(model.feature_dim, 10).to(device)
    classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.bias.data.zero_()

    # Data
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    data_dir = hydra.utils.to_absolute_path(args.data_dir)
    train_set = CIFAR10(data_dir, True, transform=train_transform)
    train_loader = DataLoader(train_set, 128, shuffle=True, pin_memory=True, num_workers=8)
    test_set = CIFAR10(data_dir, False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_set, 128, shuffle=False, pin_memory=True, num_workers=8)

    # Optimizer and scheduler
    optimizer = torch.optim.SGD(classifier.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):

        hits = 0
        mean_loss = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = model.encoder(x)
            y_hat = classifier(z)
            loss = criterion(y_hat, y)
            hits += int((y_hat.argmax(-1) == y).sum())
            mean_loss += loss.item() / len
            loss.backward()
            optimizer.step()
            scheduler.step()

        acc = hits / len(train_set) * 100
        print(f'Epoch {epoch+1}/100 | Last loss: {loss.item():.4f} | Acc: {acc:.2f}%')

    hits = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            y_hat = classifier(model.encoder(x))
        hits += int((y_hat.argmax(dim=-1) == y).sum())

    accuracy = hits / len(test_set) * 100
    print(f'Test accuracy: {accuracy:.2f}%')


@hydra.main(config_name="simclr_config.yaml", config_path=str(Path.cwd()))
def main(args: DictConfig):
    evaluate(args)


if __name__ == "__main__":
    main()
