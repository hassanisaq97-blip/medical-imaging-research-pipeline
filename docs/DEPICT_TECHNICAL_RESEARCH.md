# DEPICT-RH technical research notes

This document records what was directly observed in DEPICT-RH's public GitHub
repositories on 2026-09-06, what this project adopts as design inspiration,
what is this project's own independent implementation, and what remains
unknown because it is not publicly documented. It is written before any
project code was implemented, per the working process for this project.

Repositories inspected (public, read-only, shallow clone):

- `DEPICT-RH/Multimodal-HC` (commit `b8e04e5`)
- `DEPICT-RH/CTlessPET`
- `DEPICT-RH/bat-seg`
- `DEPICT-RH/postoperative_brain_tumor_segmentation_with_brats`

No code from these repositories is vendored or copied into this project.
They are used only as technical reference for data layout, naming
conventions, and tooling choices.

## 1. `Multimodal-HC` — the primary dataset

### 1.1 What the dataset is

A multimodal total-body dynamic ¹⁸F-FDG PET/CT/MRI dataset of 100 healthy
adults (18–100 years), stratified by age and sex, acquired on a Siemens
Biograph Vision Quadra. It is described as normative reference data for
quantitative PET research (Scientific Data, DOI not yet assigned in the
repo — placeholder `10.xxxx/xxxxxxx` in `README.md`).

### 1.2 Data access model (observed directly)

The repository itself does **not** contain raw or derived images. It
contains only code, documentation, and (per `.gitignore`) a `readouts/`
folder of small precomputed tabular files that is excluded from git and
distributed separately. Three access tiers are described in the README:

1. **Precomputed readouts** (TACs, static SUV/SUL, Patlak Ki, participant
   metadata) — described as directly downloadable/explorable, no DUA
   mentioned for these tabular derivatives. An interactive explorer is
   published at `https://hedypet.depict.dk`.
2. **Full NIfTI imaging data (PET/CT/MRI)** — distributed via a Hugging
   Face dataset repo (`huggingface.co/datasets/DEPICT-RH/Multimodal-HC`),
   gated behind signing a Data User Agreement through PublicNeuro
   (`datacatalog.publicneuro.eu/dataset/super/V2`). This is a manual,
   identity-verified application process — it cannot be automated.
3. **Raw listmode PET data (`.ptd`)** — distributed via a separate DOI
   (`doi.org/10.70883/JZJH3431`) for groups that want to redo
   reconstruction.

Only 80 of the 100 subjects (train/validation) are currently released;
the remaining 20 are held back for a 2026 MICCAI "BIC-MAC" challenge.

**Conclusion for this project:** full PET/CT/MRI NIfTI access requires a
manual DUA step that cannot and must not be bypassed. `scripts/setup_data.py`
in this project automates everything up to that point (checking for a
local `DATASET_ROOT`, printing the exact PublicNeuro/Hugging Face URLs and
what to do) and resumes automatically once the data is present locally —
see [`docs/data_access.md`](data_access.md).

### 1.3 Directory layout and naming conventions (observed directly)

The dataset follows a BIDS-like layout, confirmed by `utils.py` and the
numbered pipeline scripts in `src/multimodal_hc/scripts/`:

```
<DATASET_ROOT>/
  participants.tsv
  sub-000/ .. sub-099/
    ses-quadra/
      pet/   *_pet.nii.gz (+ .json BIDS sidecar)
      ct/    *_ct.nii.gz
      mri/   *_mri.nii.gz (DIXON in/out/water/fat, T1 MPRAGE)
  derivatives/
    totalsegmentator/<sub>/**/ct/*.nii.gz
    synthseg/<sub>/**/*.nii.gz
    nifti_dynamic/...
    pet_norm_consts/<sub>/*.txt
    registration_matrices/<sub>/mr2petct_head.txt
    readouts/{means.csv, metadata.csv}   (git-ignored, distributed separately)
```

Key filename tokens observed in `README.md` usage examples and
`02_make_pipeline_bodystat.py`:

- `*acstatPSF*_pet.nii.gz` — static PET reconstruction (PSF+TOF).
- `*acdynPSF*_pet.nii.gz` — dynamic PET reconstruction (69 time frames).
- `*br38f_ct.nii.gz` — low-dose CT, `br38f` reconstruction kernel.
- `*seg-total*.nii.gz` under `derivatives/totalsegmentator/<sub>/**/ct/` —
  TotalSegmentator "total" task labels on CT.
- Each `.nii.gz` has a same-named `.json` BIDS sidecar
  (`load_sidecar()` just swaps the extension).

Every image is a standard NIfTI-1 (`.nii.gz`), loaded with `nibabel`. This
directly justifies this project's choice of NIfTI + `nibabel`/MONAI as the
primary image path, and BIDS-style sidecars as the metadata convention.

### 1.4 Segmentation derivatives (observed directly)

Documented in `src/multimodal_hc/scripts/00_segmentation_and_defacing.md`:

- **TotalSegmentator v2.5.0** was run on CT (`ac` and `br38f`
  reconstructions) and on DIXON body MRI (`IN`/`OUT`/`W`) for the `total`,
  `tissue_types`, and `body` tasks.
- **SynthSeg** was run on MPRAGE T1 for brain regions, resampled to MPRAGE
  space with **nearest-neighbour interpolation**.
- A separate `face` task from TotalSegmentator was used only to build
  defacing masks (not scientifically relevant labels).
- Resampling of every derivative mask onto a target image uses
  `resample_and_save_bids(..., order=0)` — i.e. **nearest-neighbour
  interpolation for every label volume**, confirmed directly in
  `02_make_pipeline_bodystat.py` and `03_make_pipeline_bodydyn.py`. This is
  the direct precedent this project follows in
  `src/medimg_pipeline/preprocessing`: **never use linear/spline
  interpolation on a segmentation mask.**
- Original (non-defaced) images and face masks are withheld for privacy;
  only defaced images are released. TotalSegmentator's own public
  documentation (not re-derived here) lists the `total` task label set,
  which includes large solid organs (liver, spleen, kidneys, lungs, etc.),
  skeletal structures, and vasculature.

### 1.5 What is NOT known (unavailable without the DUA)

- Actual per-subject foreground voxel counts, image shapes, voxel
  spacing, or affine matrices — these require the real NIfTI files, which
  are gated behind the PublicNeuro DUA. This project's curation pipeline
  (`medimg_pipeline curate`) is built to compute and report these
  automatically the moment `DATASET_ROOT` points at real data; until then,
  any numbers in this repository's README/docs are explicitly labelled as
  synthetic/smoke-test results, never fabricated real ones.
- The final published DOI/citation for the dataset paper (placeholder in
  their own README).
- Exact SynthSeg version used (README says "SynthSeg vXXX").

### 1.6 Dependencies and tooling (observed directly, `pyproject.toml` /
`environment.yml`)

`nibabel`, `totalsegmentator==2.5.0` (2.14.0 in the conda env export),
`antspyx` (registration), `dcm2bids4ct`, `nifti-dynamic`, `pydicom`,
`dicom2nifti`, `nnunetv2` / `dynamic-network-architectures` (TotalSegmentator's
own nnU-Net backend), `python-dotenv` for a `DATASET_ROOT` environment
variable read from a `.env` file, Python 3.11–3.12, packaged with
`hatchling` and installed via `uv sync` or `conda env create`.

This project adopts the same `python-dotenv` + `.env` convention for
configuring data roots (see `.env.example`), and the same "numbered
pipeline scripts + BIDS derivatives" mental model for its own curation
pipeline, while implementing its own code from scratch.

**Licensing:** `Multimodal-HC` carries no `LICENSE` file and no `license`
field in `pyproject.toml` — its code is therefore all-rights-reserved by
default. This project does not copy any of its code; it only reuses the
publicly documented *data layout* as a reference so that a
`MultimodalHCLoader` in this project can locate files once a user has
their own legitimately obtained copy of the dataset.

## 2. `CTlessPET` — synthetic CT from PET (Apache-2.0)

A separately pip-installable package (`pip install CTlessPET`) that
predicts a synthetic attenuation-correction CT directly from
non-attenuation-corrected PET, for Siemens Biograph Vision/Quadra
scanners. Notable design points adopted here as CLI conventions:

- A single console-script entry point with `-i/--input`, `--output`,
  optional `--model`, `--batch_size`, `--fast` flags — the same flat,
  explicit-flag CLI style this project uses for
  `medimg_pipeline curate|anonymize|train|infer`.
- Accepts either a raw DICOM folder or NIfTI directly, auto-detecting
  which. This project's `infer` command follows the same "accept either,
  detect automatically" approach only for NIfTI in v1 (DICOM input is
  routed through the explicit `ingest` step first, see §6 of the README).

Apache-2.0 licensed; not vendored, cited in this project's README
acknowledgements.

## 3. `bat-seg` — TotalSegmentator fork adding a BAT task (Apache-2.0)

A fork of `wasserth/TotalSegmentator` adding a brown-adipose-tissue
segmentation task, trained with nnU-Net (`nnunetv2`,
`dynamic-network-architectures` — the same nnU-Net backend
TotalSegmentator itself uses). Notable structural conventions adopted here:

- Professional repo scaffolding: `.pre-commit-config.yaml`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `.github/` CI workflows, a
  `Dockerfile`, and a `tests/` directory sitting alongside the package —
  this project mirrors that shape (`pre-commit`, `CHANGELOG`-style commit
  messages, GitHub Actions CI, Docker, `tests/`).
- Model weights are downloaded lazily on first use into a hidden
  `~/.totalsegmentator` cache rather than committed to git — this project
  follows the same principle for its own trained checkpoints (never
  committed; `models/` is git-ignored, checkpoints are produced locally by
  `train`).

Apache-2.0 licensed; not vendored, cited in acknowledgements. This project
does not depend on TotalSegmentator or nnU-Net directly (that would be a
heavy, GPU-oriented dependency); it implements its own MONAI/PyTorch U-Net
instead, sized for the stated hardware constraints (CPU/MPS/CUDA on a
laptop-class machine).

## 4. `postoperative_brain_tumor_segmentation_with_brats` (Apache-2.0)

A small, single-script utility that merges a BraTS-style multi-label
tumour segmentation with an HD-GLIO segmentation into a clinically
simpler two-label scheme (tumour vs. resection cavity), with morphological
cluster-size filtering. Relevant as a general lesson for label-scheme
design (fewer, well-justified labels beat many noisy ones), which informs
§5 below. Not used directly: this project's imaging domain (whole-body
PET/CT/MRI) is unrelated to postoperative brain tumour MRI, so no code or
label scheme from this repository is reused.

## 5. Segmentation target selection — reasoning

`docs/segmentation_target_selection.md` records the full reasoning. In
short: TotalSegmentator's publicly documented `total` task (used to
produce `Multimodal-HC`'s CT-derived masks, §1.4) includes large solid
organs such as **liver**, **spleen**, and **kidneys**, alongside skeletal
and vascular structures. Given the constraints in this brief (feasible on
a laptop, sufficient foreground voxels, stable single-structure anatomy),
this project targets **liver** as its segmentation class. Multimodal-HC's
full imaging data requires a Data User Agreement (§1.2), which was not
completed in this environment, so this reasoning was never checked
against real Multimodal-HC foreground-voxel counts. The project's actual
real-data experiment instead trains and evaluates on **3D-IRCADb-01**,
which already ships its own liver ground truth directly -- see
`docs/data_access.md` for the full explanation and `docs/
segmentation_target_selection.md` for how the two relate.

## 6. Summary of what this project builds vs. what is DEPICT-RH's

| | DEPICT-RH repos | This project |
|---|---|---|
| Dataset (images, masks, readouts) | Owns and distributes | Consumes only, never redistributes |
| TotalSegmentator / SynthSeg label-generation code | Owns | Not used; masks are treated as ground truth input |
| CTlessPET / bat-seg models | Owns, Apache-2.0 | Not used; cited as inspiration only |
| BIDS-style layout, naming, nearest-neighbour mask resampling | Established convention (observed) | Adopted convention, independent implementation |
| DICOM de-identification module | Not published by DEPICT-RH | This project's own implementation (`pydicom`-based) |
| Curation manifest / QC / train-val-test splitting | Not published by DEPICT-RH | This project's own implementation |
| 3D U-Net training/inference, job queue, API, Docker | Not published by DEPICT-RH | This project's own implementation |
