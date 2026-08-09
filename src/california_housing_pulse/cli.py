"""Command-line entry point: ``chp <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .paths import ensure_dirs, relative


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


def _cmd_build(args: argparse.Namespace) -> int:
    from .data.pipeline import build

    result = build()
    if not result.validation.ok:
        print(
            f"\nERROR: {len(result.validation.errors)} data-quality check(s) failed; "
            f"see {result.report_path}",
            file=sys.stderr,
        )
        return 1
    print(f"\nPanel rebuilt: {len(result.panel):,} rows. See {result.report_path}")
    return 0


def _cmd_eda(args: argparse.Namespace) -> int:
    from .eda import build_eda
    from .io import read_parquet
    from .paths import PROCESSED_DIR

    panel_path = PROCESSED_DIR / "county_month_panel.parquet"
    if not panel_path.exists():
        print(
            f"\nERROR: {relative(panel_path)} not found. Run `chp build` first.",
            file=sys.stderr,
        )
        return 1

    print("Rendering exploratory analysis …")
    report_path, figure_paths = build_eda(read_parquet(panel_path))
    for path in figure_paths:
        print(f"  figure {path}")
    print(f"  report {report_path}")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    """Run the test suite in-process so `chp all` needs no extra tooling."""
    import pytest

    from .paths import PROJECT_ROOT

    return int(pytest.main([str(PROJECT_ROOT / "tests"), "-q"]))


def _cmd_all(args: argparse.Namespace) -> int:
    """The full documented rebuild: fetch, build, eda, test."""
    steps = (
        ("fetch", _cmd_fetch, argparse.Namespace(source=None, force=False)),
        ("build", _cmd_build, argparse.Namespace()),
        ("eda", _cmd_eda, argparse.Namespace()),
        ("test", _cmd_test, argparse.Namespace()),
    )
    for name, handler, step_args in steps:
        print(f"\n=== chp {name} " + "=" * (60 - len(name)))
        code = handler(step_args)
        if code != 0:
            print(f"\nERROR: step '{name}' failed with exit code {code}.", file=sys.stderr)
            return code
    print("\nAll steps completed.")
    return 0


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

    build_cmd = subparsers.add_parser(
        "build", help="rebuild staged tables, the joined panel, and the data-quality report"
    )
    build_cmd.set_defaults(func=_cmd_build)

    eda_cmd = subparsers.add_parser(
        "eda", help="render the exploratory analysis report and figures"
    )
    eda_cmd.set_defaults(func=_cmd_eda)

    test_cmd = subparsers.add_parser("test", help="run the test suite")
    test_cmd.set_defaults(func=_cmd_test)

    all_cmd = subparsers.add_parser("all", help="fetch, build, eda, then test — the full rebuild")
    all_cmd.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
