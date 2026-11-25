"""Tests for the deterministic catalog / query split."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from sop_search.split import (
    QUERY_FRACTION,
    ROLE_CATALOG,
    ROLE_DUPLICATE,
    ROLE_QUERY,
    Photo,
    assign_roles,
    manifest_bytes,
    read_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "manifest.csv"


def synthetic_photos(n_super: int = 3, n_classes: int = 60, seed: int = 0) -> list[Photo]:
    """Fake dataset: products with 1..6 photos, plus a few exact duplicates."""
    rng = random.Random(seed)  # only builds the fixture; the split itself uses no RNG
    photos: list[Photo] = []
    image_id = 1
    for class_id in range(1, n_classes + 1):
        super_id = class_id % n_super + 1
        for k in range(rng.randint(1, 6)):
            digest = hashlib.sha256(f"img-{image_id}".encode()).hexdigest()
            photos.append(Photo(image_id, class_id, super_id, f"c{class_id}/{k}.JPG", digest))
            image_id += 1
    # Byte-identical copies: same product, and one copy across products.
    for src_idx, class_id in [(0, photos[0].class_id), (5, photos[5].class_id), (10, 999)]:
        src = photos[src_idx]
        photos.append(Photo(image_id, class_id, src.super_class_id, f"dup/{image_id}.JPG", src.file_sha256))
        image_id += 1
    return photos


def test_split_is_independent_of_input_order():
    photos = synthetic_photos()
    shuffled = photos[:]
    random.Random(42).shuffle(shuffled)
    assert manifest_bytes(photos, assign_roles(photos)) == manifest_bytes(
        shuffled, assign_roles(shuffled)
    )


def test_duplicates_point_to_smallest_id_and_are_excluded():
    photos = synthetic_photos()
    roles = assign_roles(photos)
    by_hash = defaultdict(list)
    for p in photos:
        by_hash[p.file_sha256].append(p.image_id)
    for ids in by_hash.values():
        canonical = min(ids)
        assert roles[canonical].role in (ROLE_CATALOG, ROLE_QUERY)
        for other in ids:
            if other != canonical:
                assert roles[other].role == ROLE_DUPLICATE
                assert roles[other].duplicate_of == canonical
    assert sum(r.role == ROLE_DUPLICATE for r in roles.values()) == 3


def test_at_most_one_query_per_product_and_product_stays_in_catalog():
    photos = synthetic_photos()
    roles = assign_roles(photos)
    query_per_class = Counter()
    catalog_per_class = Counter()
    for p in photos:
        if roles[p.image_id].role == ROLE_QUERY:
            query_per_class[p.class_id] += 1
        elif roles[p.image_id].role == ROLE_CATALOG:
            catalog_per_class[p.class_id] += 1
    assert query_per_class, "split produced no queries"
    for class_id, n in query_per_class.items():
        assert n == 1
        assert catalog_per_class[class_id] >= 1


def test_query_share_per_super_category():
    photos = synthetic_photos()
    roles = assign_roles(photos)
    unique = Counter()
    queries = Counter()
    for p in photos:
        role = roles[p.image_id].role
        if role != ROLE_DUPLICATE:
            unique[p.super_class_id] += 1
        if role == ROLE_QUERY:
            queries[p.super_class_id] += 1
    for super_id, n in unique.items():
        assert queries[super_id] == round(QUERY_FRACTION * n)


@pytest.mark.skipif(not MANIFEST.exists(), reason="run scripts/build_manifest.py first")
def test_real_manifest_is_reproducible():
    """Recomputing roles from the saved manifest gives byte-identical output."""
    data = MANIFEST.read_bytes()
    photos, roles = read_manifest(MANIFEST)
    assert manifest_bytes(photos, roles) == data  # encoding round-trips
    assert manifest_bytes(photos, assign_roles(photos)) == data  # split is deterministic
