from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from sop_search.embedding import DEFAULT_MODEL, MODELS, Embedder
from sop_search.split import ROLE_CATALOG, prototype_subset, read_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute catalog embeddings")
    parser.add_argument("--subset", choices=["prototype", "full"], required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "Stanford_Online_Products"
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-catalog", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    out_dir: Path = args.out_dir or ROOT / "artifacts" / args.subset

    photos, roles = read_manifest(args.manifest)
    if args.subset == "prototype":
        catalog, _ = prototype_subset(photos, roles, args.min_catalog)
    else:
        catalog = [p for p in photos if roles[p.image_id].role == ROLE_CATALOG]
    catalog.sort(key=lambda p: p.image_id)
    print(f"{args.subset}: {len(catalog)} catalog photos, model {args.model}")

    emb = Embedder(args.model)
    ids = np.array([p.image_id for p in catalog], dtype=np.int64)
    paths = [args.dataset_dir / p.path for p in catalog]

    t0 = time.perf_counter()
    chunks = []
    step = args.batch_size * 50
    for start in range(0, len(paths), step):
        chunks.append(emb.embed_paths(paths[start : start + step], args.batch_size))
        done = min(start + step, len(paths))
        elapsed = time.perf_counter() - t0
        print(f"  {done}/{len(paths)}  {elapsed:.0f}s  eta {elapsed / done * (len(paths) - done):.0f}s", flush=True)
    vecs = np.concatenate(chunks, axis=0).astype(np.float32)
    elapsed = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", vecs)
    np.save(out_dir / "ids.npy", ids)
    meta = {
        "subset": args.subset,
        "model": args.model,
        "repo_id": emb.spec.repo_id,
        "revision": emb.spec.revision,
        "dim": int(vecs.shape[1]),
        "count": int(vecs.shape[0]),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "embed_seconds": round(elapsed, 1),
        "torch_threads": torch.get_num_threads(),
    }
    (out_dir / "embeddings.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
