# Data access

This project's real-data experiment uses **3D-IRCADb-01** (IRCAD,
France): 20 CT volumes with liver segmentation masks, widely used as a
liver-segmentation benchmark. DEPICT-RH's **Multimodal-HC** dataset was
investigated first (see `docs/DEPICT_TECHNICAL_RESEARCH.md`) but its full
PET/CT/MRI imaging data requires a Data User Agreement (DUA); IRCAD needs
only a free registration, so it was chosen as the dataset this project
actually trains and evaluates on. Multimodal-HC-related code is kept
where it is generic and independently useful (NIfTI I/O, curation,
DICOM de-identification), but **Multimodal-HC was not used to train or
evaluate the model in this project.**

## Network access in this development environment

Neither dataset's files were downloaded by this project's own tooling.
While preparing this project, the development environment's outbound
network access was found to allow only a fixed allowlist (GitHub, PyPI,
npm, Anthropic's API) -- confirmed by testing several unrelated sites
(`ircad.fr`, `huggingface.co`, `datacatalog.publicneuro.eu`,
`wikipedia.org`, `zenodo.org`), all of which were blocked identically.
This is an environment-level restriction, independent of any dataset's
own license or access terms, and it is why this project ships an
**import** command rather than a **download** command (see below): the
one manual step is downloading the data through a browser, on a machine
with normal internet access; everything after that is automated.

## 3D-IRCADb-01

- Source: IRCAD (`https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/`).
- Access: free registration, no payment; no DUA. (Registration itself was
  not completed by this project -- see the network note above.)
- Reported contents (from the wider literature and toolkits that use this
  dataset, not independently re-verified against the current official
  release in this environment): 20 patients, each with a CT DICOM series
  and one or more organ segmentation masks as parallel DICOM series,
  including a liver mask.

### Import workflow

```bash
# 1. Register and download 3D-IRCADb-01 from IRCAD yourself, on a machine
#    with normal internet access. Unzip it somewhere, e.g. ~/Downloads/3Dircadb1.

# 2. Import into this project's curated layout (de-identifies, converts
#    DICOM -> NIfTI, validates CT/mask alignment, one command):
medimg_pipeline data-import --dataset ircad \
    --input ~/Downloads/3Dircadb1 \
    --staging data/.staging/ircad \
    --output data/ircad

# 3. Curate: manifest + QC + leak-free splits.
medimg_pipeline curate --data-root data/ircad --min-foreground-voxels 500

# 4. Train / evaluate as usual (see README).
```

`data-import` discovers every directory containing a `PATIENT_DICOM`
subfolder, so it does not depend on the exact top-level folder naming
IRCAD ships with. See `medimg_pipeline.curation.ircad_import` for exactly
what it assumes and how a mismatch is reported (a clear per-patient skip
reason, not a silent wrong result).

## Multimodal-HC (investigated, not used for training)

Full PET/CT/MRI imaging data requires applying at
[datacatalog.publicneuro.eu/dataset/super/V2](https://datacatalog.publicneuro.eu/dataset/super/V2)
and signing a Data User Agreement -- a manual, identity-verified process
operated by PublicNeuro/DEPICT-RH. This project does not attempt to
automate or bypass it. Precomputed readouts (SUV/SUL/TAC tables, no
images) are available without a DUA and explorable at
[hedypet.depict.dk](https://hedypet.depict.dk). `python -m medimg_pipeline
data-check` checks for a local Multimodal-HC checkout and prints these
steps if one isn't found; it is independent of the IRCAD workflow above.

## Never committed to this repository

Per `.gitignore`: `/data/`, any `*.nii`/`*.nii.gz`/`*.dcm` file outside
`tests/fixtures/`, model checkpoints, and anything under `/outputs/`. No
IRCAD or Multimodal-HC file, in any form, is committed to this
repository's git history, a Docker image layer, or a CI artifact. CI and
the automated test suite use only procedurally generated synthetic
fixtures (`tests/helpers.py`).
