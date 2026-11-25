from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sop_search.embedding import MODELS, Embedder
from sop_search.split import prototype_subset, read_manifest

ROOT = Path(__file__).resolve().parents[1]
K = 5


def top_k(index_vecs: np.ndarray, query_vecs: np.ndarray, k: int) -> np.ndarray:
    scores = query_vecs @ index_vecs.T
    pos = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, pos, axis=1), axis=1, kind="stable")
    return np.take_along_axis(pos, order, axis=1)


def evaluate(name: str, catalog, queries, dataset_dir: Path, batch_size: int) -> dict:
    t0 = time.perf_counter()
    emb = Embedder(name)
    load_s = time.perf_counter() - t0

    paths = [dataset_dir / p.path for p in catalog]
    emb.embed_paths(paths[:batch_size], batch_size)
    t0 = time.perf_counter()
    cat_vecs = emb.embed_paths(paths, batch_size)
    cat_s = time.perf_counter() - t0
    q_vecs = emb.embed_paths([dataset_dir / p.path for p in queries], batch_size)

    cat_ids = np.array([p.image_id for p in catalog])
    cat_cls = np.array([p.class_id for p in catalog])

    self_pos = top_k(cat_vecs, cat_vecs, K)
    self_top1 = float(np.mean(cat_ids[self_pos[:, 0]] == cat_ids))

    q_pos = top_k(cat_vecs, q_vecs, K)
    q_cls = np.array([p.class_id for p in queries])
    hits = (cat_cls[q_pos] == q_cls[:, None]).any(axis=1)
    top1 = cat_cls[q_pos[:, 0]] == q_cls

    return {
        "model": name,
        "repo_id": MODELS[name].repo_id,
        "revision": MODELS[name].revision,
        "dim": emb.dim,
        "params_m": round(sum(p.numel() for p in emb.model.parameters()) / 1e6, 1),
        "load_s": round(load_s, 1),
        "ms_per_image": round(1000 * cat_s / len(catalog), 1),
        "self_top1": self_top1,
        "hit_at_5": round(float(hits.mean()), 3),
        "top1_same_product": round(float(top1.mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare embedding models on the prototype subset")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "Stanford_Online_Products"
    )
    parser.add_argument("--models", nargs="+", default=sorted(MODELS))
    parser.add_argument("--min-catalog", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "prototype" / "compare.json")
    args = parser.parse_args()

    photos, roles = read_manifest(args.manifest)
    catalog, queries = prototype_subset(photos, roles, args.min_catalog)
    n_products = len({p.class_id for p in catalog})
    print(f"prototype: {len(catalog)} catalog, {len(queries)} queries, {n_products} products")
    print(f"torch threads: {torch.get_num_threads()}")

    results = []
    for name in args.models:
        r = evaluate(name, catalog, queries, args.dataset_dir, args.batch_size)
        print(json.dumps(r))
        results.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "catalog": len(catalog),
                "queries": len(queries),
                "products": n_products,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
