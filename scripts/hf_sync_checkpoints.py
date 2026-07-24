#!/usr/bin/env python3
"""Upload/download ignored checkpoint files through a Hugging Face Dataset repo.

GitHub should keep code small; this project ignores ``*.pt``. Use this helper to
mirror the local ``checkpoint/`` directory to a Hugging Face Hub dataset repo,
then download it again on Kaggle before running E1 experiments.

Examples:
    python scripts/hf_sync_checkpoints.py upload --repo-id USER/latent-landmarks-checkpoints
    python scripts/hf_sync_checkpoints.py download --repo-id USER/latent-landmarks-checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _hub():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print(
            "ERROR: missing huggingface_hub. Install it with:\n"
            "  pip install huggingface_hub\n",
            file=sys.stderr,
        )
        sys.exit(2)
    return HfApi, snapshot_download


def _pt_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pt") if p.is_file())


def upload(args) -> None:
    HfApi, _ = _hub()
    root = Path(args.checkpoint_dir)
    files = _pt_files(root)
    if not files:
        print(f"ERROR: no .pt files found under {root}", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=args.token)
    if args.create_repo:
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True,
                        private=args.private)

    print(f"Uploading {len(files)} checkpoint file(s) from {root} to dataset {args.repo_id}")
    for path in files:
        rel = path.relative_to(root).as_posix()
        print(f"  {path} -> {rel}", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    print("Upload complete.")


def download(args) -> None:
    _, snapshot_download = _hub()
    dest = Path(args.checkpoint_dir)
    dest.mkdir(parents=True, exist_ok=True)
    patterns = args.allow_patterns or ["*.pt", "**/*.pt"]
    print(f"Downloading checkpoint files from dataset {args.repo_id} to {dest}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
        token=args.token,
    )
    files = _pt_files(dest)
    print(f"Downloaded/found {len(files)} .pt file(s):")
    for p in files:
        print(f"  {p}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-id", required=True,
                        help="Hugging Face dataset repo, e.g. USER/latent-landmarks-checkpoints")
    common.add_argument("--checkpoint-dir", default="checkpoint")
    common.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HF token. Defaults to HF_TOKEN env var.")

    up = sub.add_parser("upload", parents=[common])
    up.add_argument("--create-repo", action="store_true",
                    help="Create the dataset repo if it does not exist.")
    up.add_argument("--private", action="store_true",
                    help="Create as private when used with --create-repo.")
    up.set_defaults(func=upload)

    down = sub.add_parser("download", parents=[common])
    down.add_argument("--allow-patterns", nargs="+", default=None,
                      help="Override snapshot allow_patterns.")
    down.set_defaults(func=download)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
