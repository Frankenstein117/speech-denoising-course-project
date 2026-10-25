from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.paired_dataset import PairedAudioDataset, paired_files, split_by_speaker
from src.scratch_models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spectral_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(512, device=estimate.device)
    estimate_stft = torch.stft(estimate, 512, 128, window=window, return_complex=True)
    target_stft = torch.stft(target, 512, 128, window=window, return_complex=True)
    return F.l1_loss(torch.log1p(estimate_stft.abs()), torch.log1p(target_stft.abs()))


def enhancement_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(estimate, target) + 0.5 * spectral_loss(estimate, target)


def run_epoch(model, loader, optimizer, device, scaler, train: bool) -> float:
    model.train(train)
    total = 0.0
    batches = 0
    progress = tqdm(loader, desc="train" if train else "valid", leave=False)
    for noisy, clean in progress:
        noisy, clean = noisy.to(device), clean.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.set_grad_enabled(train), amp_context:
            enhanced = model(noisy)
            loss = enhancement_loss(enhanced, clean)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        total += float(loss.detach().cpu())
        batches += 1
        progress.set_postfix(loss=f"{total / batches:.5f}")
    return total / max(batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a speech-denoising model from random initialization.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.early_stopping_patience is not None:
        config["early_stopping_patience"] = args.early_stopping_patience
    seed_everything(int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_dir = Path(config["clean_train_dir"])
    noisy_dir = Path(config["noisy_train_dir"])
    pairs = paired_files(clean_dir, noisy_dir)
    split_seed = int(config.get("split_seed", config["seed"]))
    train_pairs, valid_pairs = split_by_speaker(
        pairs, int(config["validation_speakers"]), split_seed
    )
    valid_speakers = sorted({clean.stem.split("_")[0] for clean, _ in valid_pairs})
    print(
        {
            "total_pairs": len(pairs),
            "train_pairs": len(train_pairs),
            "valid_pairs": len(valid_pairs),
            "validation_speakers": valid_speakers,
            "split_seed": split_seed,
        }
    )
    train_data = PairedAudioDataset(
        train_pairs, int(config["sample_rate"]), float(config["segment_seconds"]), True
    )
    valid_data = PairedAudioDataset(
        valid_pairs, int(config["sample_rate"]), float(config["segment_seconds"]), False
    )
    loader_options = dict(
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_options)
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_options)

    model_kwargs = config.get("model_kwargs", {})
    model = build_model(config["model"], **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best = float("inf")
    stale = 0
    start_epoch = 1
    log_path = output_dir / "training_log.csv"
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model"] != config["model"]:
            raise ValueError(
                f"Checkpoint model {checkpoint['model']} does not match config model {config['model']}"
            )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_valid_loss", checkpoint["valid_loss"]))
        stale = int(checkpoint.get("stale_epochs", 0))
        if log_path.exists():
            with log_path.open(newline="") as history_file:
                history = list(csv.DictReader(history_file))
            if history:
                losses = [float(row["valid_loss"]) for row in history]
                best = min(losses)
                best_index = max(index for index, loss in enumerate(losses) if loss == best)
                stale = len(losses) - best_index - 1
        print(
            {
                "resume_from": str(args.resume),
                "start_epoch": start_epoch,
                "historical_best_valid_loss": best,
                "stale_epochs": stale,
            }
        )

    log_mode = "a" if args.resume is not None and log_path.exists() else "w"
    with log_path.open(log_mode, newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "valid_loss", "learning_rate"])
        if log_mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            train_loss = run_epoch(model, train_loader, optimizer, device, scaler, True)
            valid_loss = run_epoch(model, valid_loader, optimizer, device, scaler, False)
            scheduler.step(valid_loss)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            log_file.flush()
            print(row)
            checkpoint = {
                "epoch": epoch,
                "model": config["model"],
                "model_kwargs": model_kwargs,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "valid_loss": valid_loss,
                "sample_rate": int(config["sample_rate"]),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }
            if valid_loss < best:
                best = valid_loss
                stale = 0
                checkpoint["best_valid_loss"] = best
                checkpoint["stale_epochs"] = stale
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                stale += 1
            checkpoint["best_valid_loss"] = best
            checkpoint["stale_epochs"] = stale
            torch.save(checkpoint, output_dir / "last.pt")
            if stale >= int(config["early_stopping_patience"]):
                print(f"Early stopping at epoch {epoch}; best validation loss={best:.6f}")
                break


if __name__ == "__main__":
    main()

