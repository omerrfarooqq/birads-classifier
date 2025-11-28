import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from config.config import config
from utils.transforms import get_transforms
from utils.metrics import plot_confusion_matrix
from utils.visualize import save_training_plot
from training.evaluator import evaluate
from models.resnet50 import get_resnet50
from models.resnet101 import get_resnet101
from models.efficientnet import get_efficientnet


def select_model(name):
    if name == "resnet50":
        return get_resnet50()
    if name == "resnet101":
        return get_resnet101()
    if name == "efficientnet":
        return get_efficientnet()
    else:
        raise ValueError("Invalid model")


def train_model():
    train_tf, test_tf = get_transforms()

    train_set = ImageFolder("data/train", transform=train_tf)
    val_set   = ImageFolder("data/val", transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=config["batch_size"])

    model = select_model(config["model_name"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_loss, _, _ = evaluate(model, val_loader)
        train_losses.append(total_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{config['checkpoint_dir']}/best_model.pth")
            print("Saved best model.")

    save_training_plot(train_losses, val_losses)


if __name__ == "__main__":
    train_model()
