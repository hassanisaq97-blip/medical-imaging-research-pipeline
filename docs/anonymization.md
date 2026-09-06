# Anonymization, pseudonymization, and metadata de-identification

These three terms get used interchangeably in casual conversation but
mean different things, and this project is careful to only claim the one
it actually implements. The full technical rationale (with the exact tag
lists) lives in the module docstring of
`src/medimg_pipeline/anonymization/dicom.py`; this page is the
plain-language summary for the README.

## The three terms

- **Metadata de-identification** -- removing or replacing identifying
  fields in a file's *structured metadata* (DICOM tags such as
  `PatientName`, `PatientBirthDate`, `InstitutionName`, free-text
  comments, private vendor tags, ...). This is a well-defined, mechanical
  operation, and **this is what `medimg_pipeline.anonymization.dicom`
  implements.**
- **Pseudonymization** -- replacing a direct identifier (`PatientID`, the
  study/series/instance UID triplet) with a consistent surrogate value
  derived from a secret salt via SHA-256, so that instances belonging to
  "the same patient" or "the same study" still link to each other without
  exposing who they are. This project does this too, but pseudonymization
  is **reversible in principle** by anyone who has both the salt and the
  original identifier (or who can brute-force a small identifier space) --
  the salt must be kept secret, is never written to disk in any audit
  report, and is not a default value baked into the source code.
- **Anonymization**, in the strict, irreversible sense often assumed in
  casual usage, is a much stronger property than either of the above and
  is **not** what this project's tooling guarantees. Specifically:
  - **Burned-in pixel annotations** (identifying text rendered directly
    into the image pixels by some scanners/secondary-capture devices) are
    invisible to metadata scrubbing entirely, because they live in the
    pixel data, not the header. A real clinical de-identification
    pipeline needs a separate pixel-level QC step (manual review, or an
    OCR/annotation-detection model) before data can be called anonymous.
    **This project does not implement that step.**
  - Rich combinations of technical metadata that are *kept* (exact
    acquisition timestamps, a distinctive scanner configuration) can in
    principle support re-identification via linkage with an external
    record, even after de-identification.

For these reasons, this project's module, CLI command, and documentation
consistently say **"de-identification,"** not "anonymization" -- even
though the DICOM standard itself uses "Patient Identity Removed" for a
related concept, and even though everyday usage often says "anonymize"
loosely.

## Where this module is actually applied

`medimg_pipeline data-import --dataset ircad` (see `docs/data_access.md`)
runs every 3D-IRCADb-01 patient's CT and liver-mask DICOM series through
this de-identification module before conversion to NIfTI -- IRCAD is
already a public, research-cleared release with no DUA, so this is
defense in depth rather than a requirement, but it means the module is
exercised on real (if already-public) DICOM data, not only synthetic
fixtures.

The Multimodal-HC dataset (see `docs/data_access.md`) was never obtained
in this environment, so this module was **not** run against it; any
mention of Multimodal-HC's own defacing/de-identification elsewhere in
this repository (`docs/DEPICT_TECHNICAL_RESEARCH.md` §1.4) describes what
DEPICT-RH did before release, not anything this project performed.

Beyond the IRCAD import path, the module is also tested directly against
small, procedurally generated **synthetic** DICOM fixtures containing
obviously fake identifiers (see `tests/helpers.make_synthetic_dicom_series`
and `tests/test_anonymization.py`), and is kept ready for ingesting other
DICOM sources (see README / `medimg_pipeline ingest`).
