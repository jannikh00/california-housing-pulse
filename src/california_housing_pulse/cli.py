"""Command-line entry point: ``chp <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .paths import ensure_dirs


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .data.fetch import FetchError, fetch_all

    print("Fetching raw sources into data/raw/ …")
    try:
        fetch_all(only=args.source, force=args.force)
    except FetchError as exc:
        print(f"\nERROR: {exc.reason}", file=sys.stderr)
        return 1
    print("Manifest written to data/raw/manifest.json")
    return 0


def _cmd_register(args: argparse.Namespace) -> int:
    from .data import manifest as manifest_mod
    from .data.fetch import register_source
    from .data.sources import load_registry

    registry = load_registry()
    entries = manifest_mod.load_manifest()
    register_source(registry[args.source], Path(args.file), entries=entries)
    manifest_mod.save_manifest(entries)
    print("Manifest updated.")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .data import manifest as manifest_mod
    from .data.sources import load_registry

    registry = load_registry()
    entries = manifest_mod.load_manifest()
    failures = 0
    for source in registry:
        ok, message = manifest_mod.verify(entries, source.source_id, source.raw_path)
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {source.source_id}: {message}")
        if not ok and source.required:
            failures += 1
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chp",
        description="California Housing Pulse data pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download raw sources and record provenance")
    fetch.add_argument("--source", action="append", help="limit to one source id (repeatable)")
    fetch.add_argument("--force", action="store_true", help="re-download even if cached")
    fetch.set_defaults(func=_cmd_fetch)

    register = subparsers.add_parser("register", help="adopt a hand-downloaded file")
    register.add_argument("source", help="source id from configs/sources.yaml")
    register.add_argument("file", help="path to the downloaded file")
    register.set_defaults(func=_cmd_register)

    verify = subparsers.add_parser("verify", help="check raw files against recorded hashes")
    verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
