#!/usr/bin/env python3
"""Automate everything possible for Multimodal-HC data access, up to the
one manual step (signing a Data User Agreement via PublicNeuro).

Usage:
    python scripts/setup_data.py
    python -m medimg_pipeline data-check   # same thing, via the main CLI

This script never downloads, scrapes, or attempts to bypass the DUA --
that would violate the dataset's access terms (see
docs/DEPICT_TECHNICAL_RESEARCH.md §1.2 and docs/data_access.md). It only:

1. Checks whether `DATASET_ROOT` is already set and points at what looks
   like a real Multimodal-HC checkout (participants.tsv + sub-* dirs).
2. If not, prints the exact URLs and steps needed to obtain access, so
   there is nothing left to figure out manually except the identity
   verification / DUA signature step itself, which only the dataset's
   own data controller can perform.
3. Once `DATASET_ROOT` is set and valid, prints a short summary (subject
   count, whether a few expected derivative folders exist) so the rest
   of the pipeline (`medimg_pipeline curate`) can be run immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

PUBLICNEURO_URL = "https://datacatalog.publicneuro.eu/dataset/super/V2"
HUGGINGFACE_URL = "https://huggingface.co/datasets/DEPICT-RH/Multimodal-HC"
LISTMODE_DOI = "https://doi.org/10.70883/JZJH3431"
EXPLORER_URL = "https://hedypet.depict.dk"


MANUAL_STEP_MESSAGE = f"""
============================================================================
 Multimodal-HC data is not yet available on this machine.
============================================================================

This project uses DEPICT-RH's Multimodal-HC dataset (see
docs/DEPICT_TECHNICAL_RESEARCH.md). Full imaging data (PET/CT/MRI) is
gated behind a Data User Agreement -- this is a one-time, manual,
identity-verified step that CANNOT be automated and this script will not
attempt to bypass it.

>>> THE ONE MANUAL STEP YOU NEED TO DO: <<<

    1. Go to: {PUBLICNEURO_URL}
    2. Apply for access and sign the Data User Agreement.
    3. Once approved, download the NIfTI data from:
       {HUGGINGFACE_URL}
       (raw listmode PET, if you want it, is separately available at
       {LISTMODE_DOI})
    4. Set DATASET_ROOT to point at the downloaded data, e.g. in .env:
         DATASET_ROOT=/path/to/Multimodal-HC/train

After that, re-run this script (or `medimg_pipeline data-check`) -- it
will detect the data automatically and the rest of the pipeline
(`medimg_pipeline curate`, `train`, `infer`) needs no further manual setup.

While you wait for DUA approval, you can still explore the dataset's
precomputed readouts (SUV/SUL/TAC tables, no DUA required) interactively
at {EXPLORER_URL}, and every part of this project's pipeline can be
exercised end-to-end right now using synthetic fixtures:

    pytest tests/ -q
============================================================================
""".strip()


def _looks_like_multimodal_hc(root: Path) -> bool:
    return root.is_dir() and (root / "participants.tsv").exists()


def check_data_access() -> bool:
    """Returns True if DATASET_ROOT is set and looks like a real checkout."""

    from medimg_pipeline.utils.env import dataset_root

    root_str = dataset_root()
    if not root_str:
        print(MANUAL_STEP_MESSAGE)
        return False

    root = Path(root_str)
    if not _looks_like_multimodal_hc(root):
        print(f"DATASET_ROOT is set to {root}, but it does not look like a Multimodal-HC checkout")
        print("(expected to find participants.tsv there).")
        print()
        print(MANUAL_STEP_MESSAGE)
        return False

    subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    derivatives = root / "derivatives"
    print(f"DATASET_ROOT looks valid: {root}")
    print(f"  {len(subjects)} subject director(y/ies) found.")
    if derivatives.is_dir():
        found_derivs = sorted(p.name for p in derivatives.iterdir() if p.is_dir())
        print(f"  derivatives/ present: {', '.join(found_derivs) if found_derivs else '(empty)'}")
    else:
        print("  derivatives/ not found -- some curation steps may have nothing to discover.")
    print()
    print("Ready. Next step:")
    print("  medimg_pipeline curate --data-root", str(root))
    return True


def main() -> int:
    ok = check_data_access()
    return 0 if ok else 0  # informational script: never a hard failure


if __name__ == "__main__":
    sys.exit(main())
