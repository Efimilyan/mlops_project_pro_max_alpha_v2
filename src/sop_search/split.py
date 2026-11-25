from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

QUERY_FRACTION = 0.10
SPLIT_SALT = "sop-split-v1"

ROLE_CATALOG = "catalog"
ROLE_QUERY = "query"
ROLE_DUPLICATE = "duplicate"

EBAY_INFO_HEADER = ["image_id", "class_id", "super_class_id", "path"]
MANIFEST_HEADER = [
    "image_id",
    "class_id",
    "super_class_id",
    "path",
    "file_sha256",
    "role",
    "duplicate_of",
]


@dataclass(frozen=True)
class Photo:
    image_id: int
    class_id: int
    super_class_id: int
    path: str
    file_sha256: str


@dataclass(frozen=True)
class Assignment:
    role: str
    duplicate_of: int | None = None


def read_ebay_info(path: Path) -> list[tuple[int, int, int, str]]:
    rows: list[tuple[int, int, int, str]] = []
    with open(path, encoding="utf-8") as f:
        header = f.readline().split()
        if header != EBAY_INFO_HEADER:
            raise ValueError(f"Unexpected header in {path}: {header}")
        for line_no, line in enumerate(f, start=2):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 4:
                raise ValueError(f"{path}:{line_no}: expected 4 fields, got {parts}")
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    return rows


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stable_rank(kind: str, key: int) -> tuple[str, int]:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{kind}:{key}".encode()).hexdigest()
    return digest, key


def assign_roles(photos: Iterable[Photo]) -> dict[int, Assignment]:
    by_id = {p.image_id: p for p in photos}

    groups: dict[str, list[int]] = defaultdict(list)
    for image_id in sorted(by_id):
        groups[by_id[image_id].file_sha256].append(image_id)

    result: dict[int, Assignment] = {}
    unique_ids: list[int] = []
    for ids in groups.values():
        canonical = ids[0]
        unique_ids.append(canonical)
        for dup in ids[1:]:
            result[dup] = Assignment(ROLE_DUPLICATE, duplicate_of=canonical)

    photos_by_class: dict[int, list[int]] = defaultdict(list)
    classes_by_super: dict[int, set[int]] = defaultdict(set)
    unique_count_by_super: dict[int, int] = defaultdict(int)
    for image_id in sorted(unique_ids):
        p = by_id[image_id]
        photos_by_class[p.class_id].append(image_id)
        classes_by_super[p.super_class_id].add(p.class_id)
        unique_count_by_super[p.super_class_id] += 1

    query_ids: set[int] = set()
    for super_id in sorted(classes_by_super):
        target = round(QUERY_FRACTION * unique_count_by_super[super_id])
        eligible = sorted(
            (c for c in classes_by_super[super_id] if len(photos_by_class[c]) >= 2),
            key=lambda c: stable_rank("class", c),
        )
        if len(eligible) < target:
            raise ValueError(
                f"super_class_id={super_id}: need {target} donor products, "
                f"only {len(eligible)} have >= 2 unique photos"
            )
        for class_id in eligible[:target]:
            query_ids.add(
                min(photos_by_class[class_id], key=lambda i: stable_rank("photo", i))
            )

    for image_id in unique_ids:
        result[image_id] = Assignment(ROLE_QUERY if image_id in query_ids else ROLE_CATALOG)
    return result


def manifest_rows(photos: Iterable[Photo], roles: dict[int, Assignment]) -> list[list[str]]:
    rows = []
    for p in sorted(photos, key=lambda p: p.image_id):
        a = roles[p.image_id]
        rows.append(
            [
                str(p.image_id),
                str(p.class_id),
                str(p.super_class_id),
                p.path,
                p.file_sha256,
                a.role,
                "" if a.duplicate_of is None else str(a.duplicate_of),
            ]
        )
    return rows


def manifest_bytes(photos: Iterable[Photo], roles: dict[int, Assignment]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(MANIFEST_HEADER)
    writer.writerows(manifest_rows(photos, roles))
    return buf.getvalue().encode("utf-8")


def read_manifest(path: Path) -> tuple[list[Photo], dict[int, Assignment]]:
    photos: list[Photo] = []
    roles: dict[int, Assignment] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != MANIFEST_HEADER:
            raise ValueError(f"Unexpected manifest header: {reader.fieldnames}")
        for r in reader:
            p = Photo(
                int(r["image_id"]),
                int(r["class_id"]),
                int(r["super_class_id"]),
                r["path"],
                r["file_sha256"],
            )
            photos.append(p)
            dup = r["duplicate_of"]
            roles[p.image_id] = Assignment(r["role"], int(dup) if dup else None)
    return photos, roles
