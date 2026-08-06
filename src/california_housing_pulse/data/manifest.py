"""Provenance manifest for raw snapshots.

``data/raw/manifest.json`` is the one committed artifact describing the raw
snapshots that are themselves too large to commit. It answers, for every input:
which URL it came from, exactly which bytes were retrieved (SHA-256), how many,
when, what the server claimed about freshness, and whether it arrived by script
or by hand.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..paths import RAW_MANIFEST, relative

SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass
class ManifestEntry:
    """Provenance for one retrieved file."""

    source_id: str
    url: str
    path: str
    sha256: str
    bytes: int
    retrieved_at: str
    acquisition: str  # "fetch" (scripted) or "register" (manual fallback)
    http_last_modified: str | None = None
    http_etag: str | None = None


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 so a 241 MB file never loads into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_manifest(path: Path | None = None) -> dict[str, ManifestEntry]:
    """Read the manifest, returning an empty mapping if it does not exist yet."""
    manifest_path = Path(path) if path else RAW_MANIFEST
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text())
    return {
        source_id: ManifestEntry(**entry) for source_id, entry in payload.get("entries", {}).items()
    }


def save_manifest(entries: dict[str, ManifestEntry], path: Path | None = None) -> Path:
    """Write the manifest with stable key ordering so diffs stay readable."""
    manifest_path = Path(path) if path else RAW_MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "entries": {source_id: asdict(entries[source_id]) for source_id in sorted(entries)},
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return manifest_path


def record(
    entries: dict[str, ManifestEntry],
    *,
    source_id: str,
    url: str,
    path: Path,
    acquisition: str,
    http_last_modified: str | None = None,
    http_etag: str | None = None,
) -> ManifestEntry:
    """Hash a retrieved file and add or replace its manifest entry."""
    entry = ManifestEntry(
        source_id=source_id,
        url=url,
        path=relative(path),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        retrieved_at=utc_now(),
        acquisition=acquisition,
        http_last_modified=http_last_modified,
        http_etag=http_etag,
    )
    entries[source_id] = entry
    return entry


def verify(entries: dict[str, ManifestEntry], source_id: str, path: Path) -> tuple[bool, str]:
    """Check a file on disk against its recorded hash.

    Returns ``(ok, message)`` rather than raising, so callers can report every
    mismatch at once instead of stopping at the first.
    """
    entry = entries.get(source_id)
    if entry is None:
        return False, "not in manifest"
    if not path.exists():
        return False, "file missing from data/raw/"
    actual = sha256_file(path)
    if actual != entry.sha256:
        return False, f"SHA-256 mismatch (recorded {entry.sha256[:12]}…, found {actual[:12]}…)"
    return True, "ok"
