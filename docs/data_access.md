# Data access

This project's primary dataset is DEPICT-RH's **Multimodal-HC**: a
multimodal total-body dynamic ¹⁸F-FDG PET/CT/MRI dataset of 100 healthy
adults. See `docs/DEPICT_TECHNICAL_RESEARCH.md` for how its repository
and layout were investigated, and
`docs/segmentation_target_selection.md` for how this project uses it.

## What is available without any application

- **Precomputed readouts** (time-activity curves, static SUV/SUL,
  Patlak Ki, participant metadata) -- explorable interactively at
  [hedypet.depict.dk](https://hedypet.depict.dk).
- **This project's own code and synthetic fixtures** -- every module in
  this repository (curation, anonymization, preprocessing, training,
  inference, QC, the job queue) can be exercised end-to-end right now
  with zero real patient data, via `pytest tests/` or
  `bash examples/quickstart.sh`.

## What requires a Data User Agreement

Full imaging data (PET/CT/MRI, as NIfTI) requires applying at
[datacatalog.publicneuro.eu/dataset/super/V2](https://datacatalog.publicneuro.eu/dataset/super/V2)
and signing a Data User Agreement (DUA). This is a manual,
identity-verified process operated by PublicNeuro/DEPICT-RH, not by this
project, and **this project does not attempt to automate, bypass, or work
around it** -- doing so would violate the dataset's access terms.

Raw listmode PET (`.ptd`) is separately available via
[doi.org/10.70883/JZJH3431](https://doi.org/10.70883/JZJH3431), for
groups that want to redo image reconstruction; this project does not use
it (it consumes already-reconstructed PET/CT/MRI NIfTI images).

Only 80 of the 100 subjects (train/validation) are currently released;
the remaining 20 are held back for the 2026 MICCAI "BIC-MAC" challenge.

## How to get set up

1. Run `python -m medimg_pipeline data-check` (or `make data-check`).
   - If `DATASET_ROOT` is not set, or does not point at a valid
     checkout, it prints the exact steps above and stops -- there is
     nothing else to automate until the DUA is signed.
   - If `DATASET_ROOT` is already valid, it prints a subject count and
     which `derivatives/` are present, and tells you to run `curate`.
2. Once you have DUA-approved access and have downloaded the data (from
   [huggingface.co/datasets/DEPICT-RH/Multimodal-HC](https://huggingface.co/datasets/DEPICT-RH/Multimodal-HC)),
   set `DATASET_ROOT` in your `.env` (copy `.env.example`) to point at
   the downloaded `train/` directory.
3. Re-run `medimg_pipeline data-check` -- it will detect the data
   automatically.
4. Run `medimg_pipeline curate --data-root "$DATASET_ROOT"` (or
   `make curate`) to build the manifest and QC report.

## Never committed to this repository

Per `.gitignore`: `data/`, any `*.nii`/`*.nii.gz`/`*.dcm` file outside
`tests/fixtures/`, model checkpoints, and anything under `outputs/`. Real
Multimodal-HC data should never end up in a git history, a Docker image
layer, or a CI artifact -- this project's CI never downloads or touches
real data; it only ever uses procedurally generated synthetic fixtures
(see `tests/helpers.py`).
