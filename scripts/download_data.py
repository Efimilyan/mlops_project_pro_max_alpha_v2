from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

URLS = {
    "gdrive": "https://drive.usercontent.google.com/download"
    "?id=1TclrpQOF_ullUP99wk_gjGN8pKvtErG8&export=download&confirm=t",
    "ftp": "ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip",
}
ARCHIVE_NAME = "Stanford_Online_Products.zip"
EXPECTED_SIZE = 3_083_860_082
EXPECTED_SHA256 = "b04fa78210e69fab050ce73939550e1cd0f96890bdd98ec9fee732c3b756ef48"
EXTRACTED_DIR = "Stanford_Online_Products"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}\n  -> {part}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        done = 0
        next_report = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                print(f"  {done / 1e9:.2f} / {EXPECTED_SIZE / 1e9:.2f} GB", flush=True)
                next_report += 250_000_000
    part.rename(dest)


def verify(archive: Path) -> None:
    size = archive.stat().st_size
    if size != EXPECTED_SIZE:
        sys.exit(f"Size mismatch for {archive}: {size} != {EXPECTED_SIZE}")
    print("Computing sha256 ...")
    digest = sha256_of(archive)
    if digest != EXPECTED_SHA256:
        sys.exit(f"sha256 mismatch for {archive}:\n  got      {digest}\n  expected {EXPECTED_SHA256}")
    print(f"OK: size {size}, sha256 {digest}")


def extract(archive: Path, data_dir: Path) -> Path:
    target = data_dir / EXTRACTED_DIR
    marker = target / ".extracted_ok"
    if marker.exists():
        print(f"Already extracted: {target}")
        return target
    if target.exists():
        print(f"Removing incomplete extraction: {target}")
        shutil.rmtree(target)
    print(f"Extracting to {data_dir} ...")
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            resolved = (data_dir / name).resolve()
            if not resolved.is_relative_to(data_dir.resolve()):
                sys.exit(f"Unsafe path in archive: {name}")
        zf.extractall(data_dir)
    marker.touch()
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, verify and extract the Stanford Online Products archive")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--source", choices=sorted(URLS), default="gdrive")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    if archive.exists():
        print(f"Archive exists: {archive}")
    else:
        download(URLS[args.source], archive)
    verify(archive)
    target = extract(archive, data_dir)
    print(f"Dataset ready: {target}")


if __name__ == "__main__":
    main()
