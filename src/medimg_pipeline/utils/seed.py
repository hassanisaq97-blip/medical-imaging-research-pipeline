"""Reproducible random seeding across python/numpy/torch.

numpy and torch are imported lazily inside `set_seed`, consistent with
`medimg_pipeline.utils.device` -- CLI commands that never touch a model
(`anonymize`, `curate`, `ingest`) should not pay torch's import cost.
Both are hard dependencies of this project (see pyproject.toml), so no
ImportError guard is needed once inside the function.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
