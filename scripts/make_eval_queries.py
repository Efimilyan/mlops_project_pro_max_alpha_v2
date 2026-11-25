from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sop_search.split import ROLE_CATALOG, ROLE_QUERY, read_manifest, stable_rank

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_NAMES = {
    1: "bicycle",
    2: "cabinet",
    3: "chair",
    4: "coffee_maker",
    5: "fan",
    6: "kettle",
    7: "lamp",
    8: "mug",
    9: "sofa",
    10: "stapler",
    11: "table",
    12: "toaster",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Hit@5 queries and expected IDs")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "queries.csv")
    args = parser.parse_args()

    photos, roles = read_manifest(args.manifest)
    catalog_by_class: dict[int, list[int]] = defaultdict(list)
    queries_by_super = defaultdict(list)
    for p in sorted(photos, key=lambda p: p.image_id):
        role = roles[p.image_id].role
        if role == ROLE_CATALOG:
            catalog_by_class[p.class_id].append(p.image_id)
        elif role == ROLE_QUERY:
            queries_by_super[p.super_class_id].append(p)

    rows = []
    for super_id in sorted(queries_by_super):
        pool = sorted(queries_by_super[super_id], key=lambda p: stable_rank("eval", p.image_id))
        for q in pool[: args.per_category]:
            expected = catalog_by_class[q.class_id]
            if not expected:
                raise SystemExit(f"query {q.image_id} has no catalog photos of its product")
            rows.append(
                {
                    "query_id": q.image_id,
                    "class_id": q.class_id,
                    "super_class_id": super_id,
                    "path": q.path,
                    "expected_ids": " ".join(map(str, expected)),
                    "reason": (
                        f"same product class_id={q.class_id} ({CATEGORY_NAMES.get(super_id, super_id)}): "
                        f"{len(expected)} other photo(s) of this listing in the catalog"
                    ),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} queries to {args.out}")


if __name__ == "__main__":
    main()
