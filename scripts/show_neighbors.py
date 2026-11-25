from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from sop_search.embedding import DEFAULT_MODEL, Embedder, load_image
from sop_search.service import DEFAULT_URL, connect, search
from sop_search.split import read_manifest

ROOT = Path(__file__).resolve().parents[1]
TILE = 224
LABEL_H = 36


def tile(img: Image.Image, lines: list[str], color: str) -> Image.Image:
    img = img.copy()
    img.thumbnail((TILE, TILE))
    canvas = Image.new("RGB", (TILE, TILE + LABEL_H), "white")
    canvas.paste(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, TILE - 1, TILE - 1], outline=color, width=4)
    for n, line in enumerate(lines):
        draw.text((4, TILE + 2 + 16 * n), line, fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a query and its 5 neighbors from Triton")
    parser.add_argument("--image-id", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.csv")
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "data" / "raw" / "Stanford_Online_Products"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    photos, roles = read_manifest(args.manifest)
    by_id = {p.image_id: p for p in photos}
    q = by_id[args.image_id]
    q_img = load_image(args.dataset_dir / q.path)
    found = search(connect(args.url), Embedder(DEFAULT_MODEL).embed_images([q_img])[0])

    tiles = [tile(q_img, [f"QUERY {q.image_id}", f"class {q.class_id}"], "blue")]
    for rank, i in enumerate(found, start=1):
        p = by_id[i]
        same = p.class_id == q.class_id
        tiles.append(
            tile(
                load_image(args.dataset_dir / p.path),
                [f"#{rank} id {i}", f"class {p.class_id} {'SAME' if same else ''}"],
                "green" if same else "red",
            )
        )
    sheet = Image.new("RGB", (TILE * len(tiles), TILE + LABEL_H), "white")
    for n, t in enumerate(tiles):
        sheet.paste(t, (n * TILE, 0))
    out = args.out or ROOT / "artifacts" / "neighbors" / f"{args.image_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"query {q.image_id} ({q.path}) -> {found}")
    for rank, i in enumerate(found, start=1):
        print(f"  #{rank} {i} class {by_id[i].class_id} {by_id[i].path}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
