from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sop_search.split import (
    ROLE_CATALOG,
    ROLE_DUPLICATE,
    ROLE_QUERY,
    Photo,
    assign_roles,
    file_sha256,
    manifest_bytes,
    read_ebay_info,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_IMAGES = 120_053
EXPECTED_CLASSES = 22_634
EXPECTED_SUPER_CLASSES = 12
MIN_CATALOG_SIZE = 100_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the catalog / query manifest")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "Stanford_Online_Products"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    info = read_ebay_info(args.dataset_dir / "Ebay_info.txt")
    n_classes = len({r[1] for r in info})
    n_super = len({r[2] for r in info})
    print(f"Ebay_info.txt: {len(info)} images, {n_classes} products, {n_super} super-categories")
    if (len(info), n_classes, n_super) != (EXPECTED_IMAGES, EXPECTED_CLASSES, EXPECTED_SUPER_CLASSES):
        sys.exit("Dataset facts do not match the official README")
    if len({r[0] for r in info}) != len(info):
        sys.exit("image_id values are not unique")

    print(f"Hashing {len(info)} files with {args.workers} threads ...")
    paths = [args.dataset_dir / r[3] for r in info]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        digests = list(pool.map(file_sha256, paths, chunksize=256))
    photos = [Photo(i, c, s, p, d) for (i, c, s, p), d in zip(info, digests, strict=True)]

    roles = assign_roles(photos)
    counts = Counter(a.role for a in roles.values())
    n_catalog, n_query, n_dup = counts[ROLE_CATALOG], counts[ROLE_QUERY], counts[ROLE_DUPLICATE]

    print("\nsuper_class_id  catalog  query  duplicate")
    per_super: dict[int, Counter] = {}
    for p in photos:
        per_super.setdefault(p.super_class_id, Counter())[roles[p.image_id].role] += 1
    for s in sorted(per_super):
        c = per_super[s]
        print(f"{s:>14}  {c[ROLE_CATALOG]:>7}  {c[ROLE_QUERY]:>5}  {c[ROLE_DUPLICATE]:>9}")
    print(f"{'total':>14}  {n_catalog:>7}  {n_query:>5}  {n_dup:>9}")

    catalog_hashes = {p.file_sha256 for p in photos if roles[p.image_id].role == ROLE_CATALOG}
    if len(catalog_hashes) != n_catalog:
        sys.exit("Catalog contains byte-identical files")
    if n_catalog < MIN_CATALOG_SIZE:
        sys.exit(f"Catalog has {n_catalog} unique photos < {MIN_CATALOG_SIZE}")

    data = manifest_bytes(photos, roles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"\nWrote {args.out} ({len(data)} bytes)")
    print(f"manifest sha256: {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
