"""Command-line entry point: `python -m medimg_pipeline <command> ...`.

Every subcommand catches this project's own `MedImgError` hierarchy and
prints a short, actionable message (plus `hint`, if any) instead of a raw
traceback, then exits with status 1 -- debugging is a first-class feature
of this project (see README §14), not an afterthought bolted onto the CLI.
Unexpected exceptions still show a traceback (via `logger.exception`) so
real bugs are not hidden, but are clearly labelled as unexpected.
"""

from __future__ import annotations

import argparse
import sys

from medimg_pipeline.exceptions import MedImgError
from medimg_pipeline.utils.logging import get_logger

logger = get_logger("cli")


def _cmd_anonymize(args: argparse.Namespace) -> None:
    from medimg_pipeline.anonymization.dicom import deidentify_directory
    from medimg_pipeline.utils.env import anonymization_salt

    salt = args.salt or anonymization_salt()
    if not salt:
        import secrets

        salt = secrets.token_hex(16)
        logger.warning(
            "No --salt given and MEDIMG_ANON_SALT is not set; generated an ephemeral "
            "salt for this run. Pseudonyms will NOT be reproducible across runs. "
            "Set MEDIMG_ANON_SALT in your .env to get stable pseudonyms."
        )

    report = deidentify_directory(args.input, args.output, salt=salt)
    report_path = f"{args.output.rstrip('/')}/_anonymization_report.json"
    report.write_json(report_path)
    print(
        f"De-identified {report.n_processed} file(s), skipped {report.n_skipped} "
        f"non-DICOM file(s), {report.n_errors} error(s). Report: {report_path}"
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    from medimg_pipeline.imaging.ingest import ingest_dicom_directory
    from medimg_pipeline.utils.env import anonymization_salt

    salt = args.salt or anonymization_salt()
    if not salt:
        import secrets

        salt = secrets.token_hex(16)
        logger.warning("No --salt given; generated an ephemeral salt (see `anonymize` command).")

    report = ingest_dicom_directory(
        args.input, args.staging, args.output, salt=salt, backend=args.backend
    )
    report_path = f"{args.output.rstrip('/')}/_ingest_report.json"
    report.write_json(report_path)
    print(f"Converted {report.n_series_found} series. Report: {report_path}")


def _cmd_curate(args: argparse.Namespace) -> None:
    from medimg_pipeline.curation.config import CurationConfig
    from medimg_pipeline.curation.manifest import run_curation

    config = CurationConfig(
        data_root=args.data_root,
        image_glob=args.image_glob,
        mask_glob=args.mask_glob,
        label_index=args.label_index,
        min_foreground_voxels=args.min_foreground_voxels,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        output_manifest=args.output_manifest,
        output_qc_report=args.output_qc_report,
    )
    df, report = run_curation(config)
    print(
        f"Curation complete: {report.n_included}/{report.n_subjects_discovered} subjects "
        f"passed QC. Manifest: {config.output_manifest} QC report: {config.output_qc_report}"
    )
    if report.n_excluded:
        print(f"{report.n_excluded} subject(s) excluded -- see the QC report for reasons.")


def _cmd_train(args: argparse.Namespace) -> None:
    from medimg_pipeline.training.config import TrainConfig
    from medimg_pipeline.training.train import run_training

    config = TrainConfig.from_yaml(args.config)
    checkpoint_path = run_training(config)
    print(f"Training complete. Best checkpoint: {checkpoint_path}")


def _cmd_infer(args: argparse.Namespace) -> None:
    from medimg_pipeline.inference.infer import run_inference

    result = run_inference(args.input, args.checkpoint, args.output, device=args.device)
    print(f"Inference complete: {result.output_mask_path}")
    print(f"Metadata: {result.metadata_path}")

    if args.qc:
        from medimg_pipeline.imaging.nifti import load_nifti
        from medimg_pipeline.qc.visual import generate_overlay_figure

        image_array, _, _ = load_nifti(args.input)
        pred_array, _, _ = load_nifti(result.output_mask_path)
        ground_truth = None
        if args.ground_truth:
            ground_truth, _, _ = load_nifti(args.ground_truth)

        qc_path = f"{args.output.rstrip('/')}/qc_overlay.png"
        generate_overlay_figure(image_array, pred_array, qc_path, ground_truth=ground_truth)
        print(f"QC overlay: {qc_path}")


def _cmd_api(args: argparse.Namespace) -> None:
    import uvicorn

    from medimg_pipeline.utils.env import api_host, api_port

    uvicorn.run(
        "medimg_pipeline.api.main:app", host=args.host or api_host(), port=args.port or api_port()
    )


def _cmd_worker(args: argparse.Namespace) -> None:
    from medimg_pipeline.queue.worker import main as worker_main

    worker_main()


def _cmd_data_check(args: argparse.Namespace) -> None:
    from scripts.setup_data import check_data_access

    check_data_access()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medimg_pipeline", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_anonymize = subparsers.add_parser("anonymize", help="De-identify a directory of DICOM files.")
    p_anonymize.add_argument("--input", required=True)
    p_anonymize.add_argument("--output", required=True)
    p_anonymize.add_argument(
        "--salt", default=None, help="Pseudonymization salt (falls back to MEDIMG_ANON_SALT)."
    )
    p_anonymize.set_defaults(func=_cmd_anonymize)

    p_ingest = subparsers.add_parser(
        "ingest", help="De-identify + convert a DICOM directory to NIfTI."
    )
    p_ingest.add_argument("--input", required=True)
    p_ingest.add_argument(
        "--staging", required=True, help="Where de-identified DICOM files are written."
    )
    p_ingest.add_argument(
        "--output", required=True, help="Where converted NIfTI files are written."
    )
    p_ingest.add_argument("--salt", default=None)
    p_ingest.add_argument("--backend", default="dicom2nifti", choices=["dicom2nifti", "dcm2niix"])
    p_ingest.set_defaults(func=_cmd_ingest)

    p_curate = subparsers.add_parser(
        "curate", help="Build a curated dataset manifest with QC and splits."
    )
    p_curate.add_argument("--data-root", required=True)
    p_curate.add_argument("--image-glob", default="{subject}/ct/*_ct.nii.gz")
    p_curate.add_argument("--mask-glob", default="{subject}/seg/*_seg-*.nii.gz")
    p_curate.add_argument("--label-index", type=int, default=None)
    p_curate.add_argument("--min-foreground-voxels", type=int, default=500)
    p_curate.add_argument("--train-frac", type=float, default=0.7)
    p_curate.add_argument("--val-frac", type=float, default=0.15)
    p_curate.add_argument("--test-frac", type=float, default=0.15)
    p_curate.add_argument("--split-seed", type=int, default=42)
    p_curate.add_argument("--output-manifest", default="outputs/curation/data_manifest.csv")
    p_curate.add_argument("--output-qc-report", default="outputs/curation/qc_report.json")
    p_curate.set_defaults(func=_cmd_curate)

    p_train = subparsers.add_parser(
        "train", help="Train the segmentation model from a YAML config."
    )
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=_cmd_train)

    p_infer = subparsers.add_parser("infer", help="Run inference on a single NIfTI image.")
    p_infer.add_argument("--input", required=True)
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--output", required=True)
    p_infer.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p_infer.add_argument("--qc", action="store_true", help="Also generate a visual QC overlay PNG.")
    p_infer.add_argument(
        "--ground-truth", default=None, help="Optional ground-truth mask for QC overlay."
    )
    p_infer.set_defaults(func=_cmd_infer)

    p_api = subparsers.add_parser("api", help="Run the FastAPI job-queue API (uvicorn).")
    p_api.add_argument("--host", default=None)
    p_api.add_argument("--port", type=int, default=None)
    p_api.set_defaults(func=_cmd_api)

    p_worker = subparsers.add_parser("worker", help="Run an RQ worker that executes queued jobs.")
    p_worker.set_defaults(func=_cmd_worker)

    p_data_check = subparsers.add_parser(
        "data-check", help="Check Multimodal-HC data availability."
    )
    p_data_check.set_defaults(func=_cmd_data_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
        return 0
    except MedImgError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception:  # noqa: BLE001
        logger.exception("Unexpected error while running command %r", args.command)
        return 1


if __name__ == "__main__":
    sys.exit(main())
