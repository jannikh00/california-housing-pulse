"""Hybrid acquisition: scripted download with a documented manual fallback.

``fetch_source`` streams a URL to ``data/raw/`` and records provenance. If the
download fails for any reason — host blocks the request, URL moves, network is
down — it raises :class:`FetchError` carrying the source's manual procedure, and
``register_source`` adopts a hand-downloaded file into the same manifest. Both
paths produce identical provenance records apart from the ``acquisition`` field.
"""

from __future__ import annotations

from pathlib import Path

import requests

from ..paths import RAW_DIR, relative
from . import manifest as manifest_mod
from .manifest import ManifestEntry
from .sources import Source, SourceRegistry, load_registry

_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_TIMEOUT_SECONDS = 60


class FetchError(RuntimeError):
    """A scripted download failed; the manual procedure is attached."""

    def __init__(self, source: Source, reason: str) -> None:
        self.source = source
        self.reason = reason
        super().__init__(reason)

    def guidance(self) -> str:
        return (
            f"\nScripted download of '{self.source.source_id}' failed: {self.reason}\n"
            f"URL: {self.source.url}\n\n"
            f"Manual fallback:\n{self.source.manual_procedure or '  (none documented)'}"
        )


def fetch_source(
    source: Source,
    *,
    user_agent: str,
    entries: dict[str, ManifestEntry],
    force: bool = False,
) -> ManifestEntry:
    """Download one source into ``data/raw/`` and record its provenance.

    A file already present with a matching recorded hash is left alone unless
    ``force`` is set, so re-running the pipeline does not re-download 241 MB.
    """
    destination = source.raw_path

    if destination.exists() and not force:
        ok, _ = manifest_mod.verify(entries, source.source_id, destination)
        if ok:
            print(f"  {source.source_id}: cached and hash-verified ({relative(destination)})")
            return entries[source.source_id]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # Download to a temporary name so an interrupted transfer never leaves a
    # truncated file that looks complete.
    partial = destination.with_suffix(destination.suffix + ".partial")

    try:
        with requests.get(
            source.url,
            headers={"User-Agent": user_agent},
            stream=True,
            timeout=_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            last_modified = response.headers.get("Last-Modified")
            etag = response.headers.get("ETag")
            total = response.headers.get("Content-Length")
            expected = int(total) if total and total.isdigit() else None

            written = 0
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_BYTES):
                    handle.write(chunk)
                    written += len(chunk)

            if expected is not None and written != expected:
                raise FetchError(
                    source, f"incomplete download: expected {expected:,} bytes, got {written:,}"
                )
    except FetchError:
        partial.unlink(missing_ok=True)
        raise
    except requests.RequestException as exc:
        partial.unlink(missing_ok=True)
        raise FetchError(source, f"{type(exc).__name__}: {exc}") from exc

    partial.replace(destination)
    entry = manifest_mod.record(
        entries,
        source_id=source.source_id,
        url=source.url,
        path=destination,
        acquisition="fetch",
        http_last_modified=last_modified,
        http_etag=etag,
    )
    print(f"  {source.source_id}: downloaded {entry.bytes:,} bytes -> {relative(destination)}")
    if last_modified:
        print(f"    server Last-Modified: {last_modified}")
    return entry


def register_source(
    source: Source,
    downloaded_file: Path,
    *,
    entries: dict[str, ManifestEntry],
) -> ManifestEntry:
    """Adopt a hand-downloaded file into ``data/raw/`` and the manifest."""
    downloaded_file = Path(downloaded_file).expanduser().resolve()
    if not downloaded_file.exists():
        raise FileNotFoundError(f"No such file: {downloaded_file}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = source.raw_path
    if downloaded_file != destination.resolve():
        destination.write_bytes(downloaded_file.read_bytes())

    entry = manifest_mod.record(
        entries,
        source_id=source.source_id,
        url=source.url,
        path=destination,
        acquisition="register",
        http_last_modified=None,
        http_etag=None,
    )
    print(f"  {source.source_id}: registered {entry.bytes:,} bytes -> {relative(destination)}")
    return entry


def fetch_all(
    *,
    registry: SourceRegistry | None = None,
    only: list[str] | None = None,
    force: bool = False,
) -> dict[str, ManifestEntry]:
    """Fetch every declared source, continuing past failures.

    Optional sources that fail are reported and skipped. Required sources that
    fail raise after all other sources have been attempted, so one broken host
    does not hide the state of the rest.
    """
    registry = registry or load_registry()
    entries = manifest_mod.load_manifest()
    wanted = only or registry.ids
    failures: list[FetchError] = []

    for source_id in wanted:
        source = registry[source_id]
        try:
            fetch_source(source, user_agent=registry.user_agent, entries=entries, force=force)
        except FetchError as exc:
            failures.append(exc)
            severity = "REQUIRED" if source.required else "optional"
            print(f"  {source_id}: FAILED ({severity}) — {exc.reason}")

    manifest_mod.save_manifest(entries)

    for failure in failures:
        print(failure.guidance())

    blocking = [f for f in failures if f.source.required]
    if blocking:
        names = ", ".join(f.source.source_id for f in blocking)
        raise FetchError(
            blocking[0].source,
            f"required source(s) unavailable: {names}. "
            "Follow the manual procedure above, then use `chp register`.",
        )
    return entries
