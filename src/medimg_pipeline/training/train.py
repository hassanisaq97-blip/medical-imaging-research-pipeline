"""Training loop for 3D liver segmentation.

Handles device selection (CUDA -> MPS -> CPU), mixed precision (CUDA
only, see `medimg_pipeline.utils.device.supports_amp`), checkpointing of
the best model by validation Dice, early stopping, and local experiment
tracking. Raises `OutOfMemoryError` with an actionable hint instead of
letting a raw CUDA/MPS OOM exception propagate.
"""

from __future__ import annotations

import time
from pathlib import Path

from medimg_pipeline.exceptions import OutOfMemoryError
from medimg_pipeline.training.config import TrainConfig
from medimg_pipeline.training.tracking import EpochRecord, ExperimentTracker
from medimg_pipeline.utils.device import resolve_device, supports_amp
from medimg_pipeline.utils.logging import get_logger
from medimg_pipeline.utils.seed import set_seed

logger = get_logger("training.train")


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def run_training(config: TrainConfig) -> Path:
    """Run training and return the path to the best checkpoint."""

    import torch
    from monai.losses import DiceCELoss
    from monai.metrics import DiceMetric
    from monai.transforms import AsDiscrete

    from medimg_pipeline.data.dataset import build_dataloaders
    from medimg_pipeline.models.unet import build_unet

    set_seed(config.seed)
    device = resolve_device(config.device)
    use_amp = config.amp and supports_amp(device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{config.experiment_name}_best.pt"

    train_loader, val_loader = build_dataloaders(
        config.manifest_path,
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = build_unet(
        channels=config.channels, strides=config.strides, num_res_units=config.num_res_units
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    tracker = ExperimentTracker(
        experiment_name=config.experiment_name,
        output_dir=output_dir,
        config=config.__dict__,
        manifest_path=config.manifest_path,
    )

    best_dice = -1.0
    epochs_without_improvement = 0

    logger.info(
        "Starting training: device=%s amp=%s epochs=%d train_batches=%d val_batches=%d",
        device,
        use_amp,
        config.num_epochs,
        len(train_loader),
        len(val_loader),
    )

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            try:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    raise OutOfMemoryError(exc) from exc
                raise

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_loss = None
        val_dice = None

        if epoch % config.val_interval == 0 and len(val_loader) > 0:
            val_loss, val_dice = _validate(
                model,
                val_loader,
                loss_fn,
                dice_metric,
                post_pred,
                post_label,
                device,
                config.patch_size,
            )

            if val_dice is not None and val_dice > best_dice:
                best_dice = val_dice
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "val_dice": val_dice,
                    },
                    checkpoint_path,
                )
                logger.info(
                    "Epoch %d: new best val_dice=%.4f -> saved %s", epoch, val_dice, checkpoint_path
                )
            else:
                epochs_without_improvement += 1

        runtime = time.time() - epoch_start
        tracker.log_epoch(
            EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_dice=val_dice,
                runtime_seconds=runtime,
                device=str(device),
            )
        )
        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%s val_dice=%s (%.1fs)",
            epoch,
            config.num_epochs,
            train_loss,
            f"{val_loss:.4f}" if val_loss is not None else "-",
            f"{val_dice:.4f}" if val_dice is not None else "-",
            runtime,
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch,
                epochs_without_improvement,
            )
            break

    if not checkpoint_path.exists():
        # No validation batches ever ran (e.g. tiny smoke-test dataset) --
        # still save something so downstream inference has a checkpoint.
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config.__dict__, "val_dice": None},
            checkpoint_path,
        )

    return checkpoint_path


def _validate(model, val_loader, loss_fn, dice_metric, post_pred, post_label, device, patch_size):
    """Validate on full-size (uncropped) volumes.

    Unlike training (which always sees fixed-size patches from
    RandCropByPosNegLabeld), a validation volume's resampled size is
    whatever it happens to be for that subject, and is not guaranteed to
    be divisible by the U-Net's total downsampling factor. Calling the
    model directly on such an input can crash with a tensor-size
    mismatch at a skip connection (odd feature-map size at some level).
    `sliding_window_inference` -- the same function `medimg_pipeline.
    inference.infer` uses -- avoids this by padding internally and
    stitching together patch-sized windows, so validation always runs the
    model at the same patch size it was trained at.
    """

    import torch
    from monai.inferers import sliding_window_inference

    model.eval()
    dice_metric.reset()
    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            try:
                outputs = sliding_window_inference(
                    images, roi_size=patch_size, sw_batch_size=1, predictor=model
                )
            except RuntimeError as exc:
                if _is_oom_error(exc):
                    raise OutOfMemoryError(exc) from exc
                raise
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            n_batches += 1

            preds = [post_pred(o) for o in torch.unbind(outputs, dim=0)]
            targets = [post_label(t) for t in torch.unbind(labels, dim=0)]
            dice_metric(y_pred=torch.stack(preds), y=torch.stack(targets))

    val_loss = running_loss / max(n_batches, 1)
    val_dice = float(dice_metric.aggregate().item()) if n_batches > 0 else None
    return val_loss, val_dice
