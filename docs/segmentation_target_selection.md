# Segmentation target selection

This project's primary ML task is automatic 3D anatomical segmentation
from CT. This document records the reasoning for the chosen target,
written before training any model, per the project brief's requirement to
inspect available labels rather than choosing arbitrarily.

## Constraints

- Must be computationally feasible to train (at least a smoke-test /
  small-scale run) on a laptop: CPU, or Apple Silicon MPS, or a modest
  CUDA GPU.
- Must have a sufficient number of positive (foreground) voxels per
  volume, so a 3D patch-based U-Net sees enough signal without extreme
  class imbalance.
- Must be anatomically stable and have a simple, well-defined boundary,
  so the label itself is not the noisiest part of the pipeline.
- Must come from a segmentation source already present in `Multimodal-HC`
  (i.e. TotalSegmentator `total`-task masks on the low-dose CT,
  `*br38f_ct.nii.gz` — see `docs/DEPICT_TECHNICAL_RESEARCH.md` §1.3–1.4),
  so no additional labelling tool needs to be introduced.

## Candidates considered

TotalSegmentator's public `total` task (as run in `Multimodal-HC`,
observed directly — see `docs/DEPICT_TECHNICAL_RESEARCH.md`) segments over
100 structures on CT, spanning organs, muscles, vasculature and the full
skeleton. Restricting to large, single-contiguous-blob solid organs —
the category most amenable to a small 3D U-Net without a hierarchical or
multi-stage approach — narrows this to:

| Candidate | Typical relative size | Anatomical stability | Notes |
|---|---|---|---|
| **Liver** | Very large (one of the largest single organs) | High — consistent shape/location across adults | Also the standard PET SUV reference organ, so segmenting it has direct downstream value for the same dataset's quantitative PET readouts |
| Spleen | Medium | High, but size varies more with age/pathology than liver | Good second target, smaller foreground → faster iteration |
| Kidneys (L+R) | Medium, bilateral | High | Two separate blobs complicates a "single foreground class" simplification |
| Lungs | Very large | High | Mostly air — segmentation is almost "easy" by intensity thresholding alone, less interesting as an ML demonstration |
| Urinary bladder | Small, variable | Low — volume varies hugely with fill state (the dataset's own README shows it has the single highest mean SUV of any organ, precisely because of this variability) | Rejected: unstable volume undermines "stable anatomy" requirement |

## Decision

**Primary target: liver, segmented from low-dose CT
(`*br38f_ct.nii.gz`), label taken from the corresponding
`derivatives/totalsegmentator/<sub>/**/ct/*seg-total*.nii.gz` volume,
binarized to the liver class.**

Rationale:
1. Large, contiguous, single-blob organ → most positive voxels of any
   single-organ candidate, minimizing class imbalance for patch sampling.
2. Stable shape and location across the 18–100-year age range covered by
   the dataset.
3. Directly reusable by the dataset's own downstream use case (PET SUV
   normalization uses the liver as a reference region), so the trained
   model has a plausible real use beyond the portfolio demonstration
   itself.
4. Binary (organ vs. background) segmentation keeps the 3D U-Net small
   enough to iterate on CPU/MPS, matching the project's hardware
   constraints, while remaining a real, non-trivial 3D volumetric task
   (unlike lungs, which are largely separable by intensity alone).

**Documented extension point, not yet built:** kidneys as a second,
multi-instance target once the liver pipeline is validated — the
curation manifest already records per-label foreground voxel counts for
every TotalSegmentator label present, not just the liver, so extending the
target list later does not require re-running curation.

## Important caveat

This decision is based on TotalSegmentator's publicly documented `total`
task label set and general medical-imaging knowledge of organ size and
stability, cross-checked against how `Multimodal-HC` itself uses the
liver (as a PET reference organ). It is **not** based on running the
curation pipeline against the real dataset, because full imaging access
requires a manual PublicNeuro Data User Agreement that has not yet been
completed in this environment (see `docs/data_access.md`). Running
`python -m medimg_pipeline curate --data-root <DATASET_ROOT>` against the
real data will produce actual per-subject, per-label foreground voxel
counts and QC flags; if that data contradicts the assumptions above (e.g.
unexpectedly poor liver mask QC for a subset of subjects), the manifest
and QC report — not this document — are authoritative, and this document
should be revisited.

All quantitative results shown elsewhere in this repository until that
point are produced on synthetic fixtures and are explicitly labelled as
such — never presented as real measurements on `Multimodal-HC`.
