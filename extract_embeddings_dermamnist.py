from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

import medmnist
from medmnist import INFO


def build_feature_extractor(model_name: str = "resnet18") -> nn.Module:
    """
    Returns a CNN that outputs embedding vectors (no final classification layer).
    """
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        # Remove final FC layer -> output is 512-d embedding
        model.fc = nn.Identity()
        preprocess = weights.transforms()
        return model, preprocess

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        # Remove final FC layer -> output is 2048-d embedding
        model.fc = nn.Identity()
        preprocess = weights.transforms()
        return model, preprocess

    raise ValueError(f"Unsupported model_name: {model_name}")


@torch.no_grad()
def extract_split_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    all_embeds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Extracting", leave=False):
        images = images.to(device)
        # labels from medmnist are usually shape (N, 1)
        labels = labels.view(-1).cpu().numpy()

        embeds = model(images).cpu().numpy()
        all_embeds.append(embeds)
        all_labels.append(labels)

    X = np.concatenate(all_embeds, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data", help="Output folder (default: data)")
    parser.add_argument("--size", type=int, default=28, help="DermMNIST image size (default: 28)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    args = parser.parse_args()

    data_flag = "dermamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model + preprocessing
    model, preprocess = build_feature_extractor(args.model)
    model = model.to(device)

    # MedMNIST gives PIL-like images if we pass transform that expects PIL,
    # but DermMNIST samples are NumPy arrays. So we wrap a transform pipeline:
    # - Convert to PIL
    # - Make 3 channels (DermMNIST is RGB already, but safe)
    # - Resize to model expected size
    # - Normalize using ImageNet stats (weights.transforms handles this)
    to_tensor_pipeline = preprocess  # includes Resize, ToTensor, Normalize, etc.

    # Datasets
    train_ds = DataClass(split="train", download=True, size=args.size, transform=to_tensor_pipeline)
    val_ds   = DataClass(split="val",   download=True, size=args.size, transform=to_tensor_pipeline)
    test_ds  = DataClass(split="test",  download=True, size=args.size, transform=to_tensor_pipeline)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print("Extracting train embeddings...")
    X_train, y_train = extract_split_embeddings(model, train_loader, device)
    print("Extracting val embeddings...")
    X_val, y_val = extract_split_embeddings(model, val_loader, device)
    print("Extracting test embeddings...")
    X_test, y_test = extract_split_embeddings(model, test_loader, device)

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dermamnist_{args.size}_{args.model}_embeddings.npz"

    np.savez_compressed(
        out_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )

    print(f"Saved embeddings: {out_path}")
    print(f"Shapes: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")


if __name__ == "__main__":
    main()
