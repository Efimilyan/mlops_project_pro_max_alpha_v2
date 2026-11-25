from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from tritonclient.utils import InferenceServerException

from sop_search.service import MODEL_NAME, connect, search
from sop_search.split import ROLE_CATALOG, read_manifest, stable_rank

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "manifest.csv"
DATASET_DIR = ROOT / "data" / "raw" / "Stanford_Online_Products"
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
DIM = 512


def _server_ready() -> bool:
    try:
        return connect(TRITON_URL).is_model_ready(MODEL_NAME)
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not MANIFEST.exists(), reason="run scripts/build_manifest.py first"),
    pytest.mark.skipif(not _server_ready(), reason=f"Triton with {MODEL_NAME} is not ready at {TRITON_URL}"),
]


@pytest.fixture(scope="module")
def manifest():
    photos, roles = read_manifest(MANIFEST)
    return {p.image_id: p for p in photos}, roles


@pytest.fixture(scope="module")
def client():
    return connect(TRITON_URL)


@pytest.fixture(scope="module")
def embedder():
    from sop_search.embedding import Embedder

    return Embedder("clip-vit-b32")


def self_retrieval_ids(by_id, roles) -> list[int]:
    by_super = defaultdict(list)
    for image_id, p in by_id.items():
        if roles[image_id].role == ROLE_CATALOG:
            by_super[p.super_class_id].append(image_id)
    return [min(ids, key=lambda i: stable_rank("selftest", i)) for _, ids in sorted(by_super.items())]


def valid_vector() -> np.ndarray:
    v = np.ones(DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_self_retrieval(manifest, client, embedder):
    by_id, roles = manifest
    ids = self_retrieval_ids(by_id, roles)
    assert len(ids) >= 10
    vecs = embedder.embed_paths([DATASET_DIR / by_id[i].path for i in ids])
    failures = []
    for image_id, vec in zip(ids, vecs, strict=True):
        found = search(client, vec)
        assert len(found) == 5
        assert all(f in by_id and roles[f].role == ROLE_CATALOG for f in found), found
        if found[0] != image_id:
            failures.append((image_id, found))
    assert not failures, f"top-1 != source for {failures}"


def test_wrong_dimension_is_rejected_then_valid_request_succeeds(client):
    with pytest.raises(InferenceServerException, match=r"Expected \[512\], got \[384\]"):
        search(client, np.ones(384, dtype=np.float32))
    found = search(client, valid_vector())
    assert len(found) == 5


def test_nan_is_rejected_then_valid_request_succeeds(client):
    bad = valid_vector()
    bad[0] = np.nan
    with pytest.raises(InferenceServerException, match="NaN"):
        search(client, bad)
    assert len(search(client, valid_vector())) == 5
