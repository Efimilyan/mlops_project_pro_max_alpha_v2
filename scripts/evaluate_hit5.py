from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sop_search.embedding import DEFAULT_MODEL, Embedder
from sop_search.service import DEFAULT_URL, connect, search
from sop_search.split import read_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hit@5 on eval/queries.csv against the running Triton")
    parser.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "Stanford_Online_Products"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    with open(args.queries, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    photos, roles = read_manifest(args.manifest)
    by_id = {p.image_id: p for p in photos}

    emb = Embedder(DEFAULT_MODEL)
    client = connect(args.url)
    vecs = emb.embed_paths([args.dataset_dir / q["path"] for q in queries])

    rows = []
    for q, vec in zip(queries, vecs, strict=True):
        qid = int(q["query_id"])
        expected = {int(x) for x in q["expected_ids"].split()}
        found = search(client, vec)
        ranks = [r for r, i in enumerate(found, start=1) if i in expected]
        rows.append(
            {
                "query_id": qid,
                "class_id": q["class_id"],
                "returned_ids": " ".join(map(str, found)),
                "returned_class_ids": " ".join(str(by_id[i].class_id) for i in found),
                "hit_at_5": int(bool(ranks)),
                "first_hit_rank": ranks[0] if ranks else "",
                "query_in_index": int(qid in found),
            }
        )
        mark = f"hit @{ranks[0]}" if ranks else "MISS"
        print(f"{qid:>7} class {q['class_id']:>6}: {found}  {mark}")

    hits = sum(r["hit_at_5"] for r in rows)
    leaks = sum(r["query_in_index"] for r in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nHit@5 = {hits}/{len(rows)} = {hits / len(rows):.3f}")
    print(f"queries returned as their own neighbor (must be 0): {leaks}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
