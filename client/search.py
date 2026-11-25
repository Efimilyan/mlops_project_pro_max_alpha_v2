from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tritonclient.utils import InferenceServerException

from sop_search.embedding import DEFAULT_MODEL, MODELS, Embedder
from sop_search.service import DEFAULT_URL, connect, search
from sop_search.split import read_manifest

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "raw" / "Stanford_Online_Products"


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the 5 most similar catalog photos")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--image-id", type=int)
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    photos, roles = read_manifest(args.manifest)
    by_id = {p.image_id: p for p in photos}
    if args.image_id is not None:
        if args.image_id not in by_id:
            sys.exit(f"unknown image_id {args.image_id}")
        query = by_id[args.image_id]
        image_path = args.dataset_dir / query.path
        print(f"query: id={query.image_id} class_id={query.class_id} role={roles[query.image_id].role} {query.path}")
    else:
        image_path = args.image
        print(f"query: {image_path}")

    vec = Embedder(args.model).embed_paths([image_path])[0]
    try:
        ids = search(connect(args.url), vec)
    except InferenceServerException as e:
        sys.exit(f"server error: {e}")

    rows = []
    for rank, i in enumerate(ids, start=1):
        p = by_id.get(i)
        rows.append(
            {
                "rank": rank,
                "image_id": i,
                "class_id": p.class_id if p else None,
                "role": roles[i].role if p else None,
                "path": str(args.dataset_dir / p.path) if p else None,
            }
        )
    print(json.dumps({"neighbor_ids": ids, "neighbors": rows}, indent=2))


if __name__ == "__main__":
    main()
