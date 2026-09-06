"""Local, privacy-safe experiment tracking.

Writes structured JSON (one file per run, containing metadata + a
per-epoch history) and a companion CSV of per-epoch metrics, so results
can be diffed in git-free tooling (Excel, pandas) without standing up any
external service. Nothing here ever leaves the local filesystem, and
nothing here contains dataset content -- only aggregate metrics, config,
and provenance (git commit, dataset manifest hash).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _file_hash(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float | None
    val_dice: float | None
    runtime_seconds: float
    device: str


@dataclass
class ExperimentTracker:
    experiment_name: str
    output_dir: Path
    config: dict
    manifest_path: str
    git_commit: str = field(default_factory=_git_commit)
    manifest_hash: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: list[EpochRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_hash = _file_hash(self.manifest_path)

    def log_epoch(self, record: EpochRecord) -> None:
        self.history.append(record)
        self._write()

    def _write(self) -> None:
        payload = {
            "experiment_name": self.experiment_name,
            "git_commit": self.git_commit,
            "manifest_path": self.manifest_path,
            "manifest_hash": self.manifest_hash,
            "config": self.config,
            "started_at": self.started_at,
            "history": [asdict(r) for r in self.history],
        }
        (self.output_dir / f"{self.experiment_name}.json").write_text(json.dumps(payload, indent=2))

        csv_path = self.output_dir / f"{self.experiment_name}.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_dice",
                    "runtime_seconds",
                    "device",
                ],
            )
            writer.writeheader()
            for r in self.history:
                writer.writerow(asdict(r))
