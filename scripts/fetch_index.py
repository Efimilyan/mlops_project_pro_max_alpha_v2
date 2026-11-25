from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RELEASE_URL = (
    "https://github.com/Efimilyan/mlops_project_pro_max_alpha_v2/releases/download/"
    "index-clip-vit-b32-v1"
)
FILES = {
    "index.faiss": "f0d37d15d7e786bd414da704bee22fcd3e3affd75a6da94ba7e5f18700c202e0",
    "index.json": "c09492adffa6a310b733f3f777d1a7052675e1c1d62f9b58d79f4b94a9677279",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, expected: str, out_dir: Path) -> None:
    dest = out_dir / name
    if dest.exists() and sha256_of(dest) == expected:
        print(f"{name}: already present, sha256 OK")
        return
    url = f"{RELEASE_URL}/{name}"
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"{name}: downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    digest = sha256_of(part)
    if digest != expected:
        part.unlink()
        sys.exit(f"{name}: sha256 mismatch\n  got      {digest}\n  expected {expected}")
    part.rename(dest)
    print(f"{name}: OK, sha256 {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the prebuilt FAISS index from the GitHub release")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "full")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        fetch(name, expected, args.out_dir)
    print(f"index ready: {args.out_dir / 'index.faiss'}")


if __name__ == "__main__":
    main()
