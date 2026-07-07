#!/usr/bin/env python3
"""Download LibriSpeech subsets and emit WhisperLite JSONL manifests.

Example (≈ 6.6 GB total for the two train-clean subsets):

    python scripts/prepare_librispeech.py --root data/librispeech \
        --subset train-clean-100 --subset dev-clean

    # produces data/librispeech/manifests/train-clean-100.jsonl, ...

The audio stays in FLAC (libsndfile decodes it directly), so no transcoding
is needed. Combine subsets by concatenating manifest files.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf

BASE_URL = "https://www.openslr.org/resources/12"
KNOWN_SUBSETS = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)


def download(subset: str, root: Path) -> Path:
    archive = root / f"{subset}.tar.gz"
    if archive.exists():
        print(f"[skip] {archive} already downloaded")
        return archive
    url = f"{BASE_URL}/{subset}.tar.gz"
    print(f"[download] {url}")
    root.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_suffix(".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    tmp.rename(archive)
    return archive


def extract(archive: Path, root: Path) -> Path:
    marker = root / f".extracted-{archive.stem}"
    if marker.exists():
        print(f"[skip] {archive.name} already extracted")
        return root / "LibriSpeech"
    print(f"[extract] {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")
    marker.touch()
    return root / "LibriSpeech"


def build_manifest(subset_dir: Path, manifest_path: Path) -> int:
    count = 0
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as out:
        for transcript_file in sorted(subset_dir.rglob("*.trans.txt")):
            for line in transcript_file.read_text(encoding="utf-8").splitlines():
                utt_id, _, text = line.partition(" ")
                flac = transcript_file.parent / f"{utt_id}.flac"
                if not flac.is_file():
                    print(f"[warn] missing {flac}", file=sys.stderr)
                    continue
                info = sf.info(str(flac))
                record = {
                    "audio_filepath": str(flac.resolve()),
                    "text": text.strip().lower(),
                    "duration": round(info.frames / info.samplerate, 3),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/librispeech", help="download/extract directory")
    parser.add_argument(
        "--subset",
        action="append",
        required=True,
        choices=KNOWN_SUBSETS,
        help="subset to fetch (repeatable)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    for subset in args.subset:
        archive = download(subset, root)
        library_root = extract(archive, root)
        subset_dir = library_root / subset
        manifest = root / "manifests" / f"{subset}.jsonl"
        count = build_manifest(subset_dir, manifest)
        print(f"[done] {subset}: {count} utterances -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
