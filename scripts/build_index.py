from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from saved embeddings")
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    d: Path = args.dir

    vecs = np.load(d / "embeddings.npy")
    ids = np.load(d / "ids.npy")
    meta = json.loads((d / "embeddings.json").read_text())
    if vecs.dtype != np.float32 or ids.dtype != np.int64 or len(vecs) != len(ids):
        sys.exit(f"bad inputs: {vecs.dtype} {vecs.shape}, {ids.dtype} {ids.shape}")
    if len(np.unique(ids)) != len(ids):
        sys.exit("ids are not unique")
    norms = np.linalg.norm(vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        sys.exit(f"embeddings are not L2-normalized: min {norms.min()}, max {norms.max()}")

    t0 = time.perf_counter()
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(vecs.shape[1]))
    index.add_with_ids(vecs, ids)
    build_s = time.perf_counter() - t0

    probe = min(len(ids), 1000)
    _, found = index.search(vecs[:probe], K)
    if not np.array_equal(found[:, 0], ids[:probe]):
        bad = int(np.sum(found[:, 0] != ids[:probe]))
        sys.exit(f"self-retrieval failed for {bad}/{probe} vectors")

    path = d / "index.faiss"
    faiss.write_index(index, str(path))
    index_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    info = {
        **meta,
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "vectors": int(index.ntotal),
        "dim": int(index.d),
        "unique_ids": int(len(np.unique(faiss.vector_to_array(index.id_map)))),
        "index_bytes": path.stat().st_size,
        "index_sha256": index_sha,
        "build_seconds": round(build_s, 2),
        "faiss_version": faiss.__version__,
    }
    (d / "index.json").write_text(json.dumps(info, indent=2))
    print(f"vectors: {info['vectors']}, dim: {info['dim']}, unique ids: {info['unique_ids']}")
    print(f"index: {path} ({info['index_bytes'] / 1e6:.1f} MB), sha256 {index_sha}")
    print(f"self top-1 on first {probe} vectors: OK")


if __name__ == "__main__":
    main()
