# data_code/parse_annotations.py
"""Walk SUN2012 Annotations/ and emit one JSON record per image.

Each LabelMe XML gives a <folder> (the SUN397-style scene path, e.g.
"b/bus_station/indoor") and a list of <object> polygons. We keep the scene
label and the set of normalised object types present -- polygons/coordinates
are irrelevant to co-occurrence.

Two things the raw XML makes easy to get wrong:
  * <deleted>1</deleted> objects are retained in the file by the LabelMe
    webtool. They are annotator retractions and must not be counted.
  * a <name> may repeat within an image (three separate "wall" polygons).
    Co-occurrence is over *types present*, so we de-duplicate per image.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

# XML 1.0 permits only tab/LF/CR below 0x20. A couple of SUN2012 files carry a
# stray 0x1A (SUB) byte and fail strict parsing; strip and retry rather than
# discarding the image.
_BAD_CTRL = bytes(b for b in range(32) if b not in (9, 10, 13))
_CTRL_TABLE = bytes.maketrans(_BAD_CTRL, b" " * len(_BAD_CTRL))

from data_code.sun_vocab import normalise
from src.paths import ANNOTATIONS_DIR, IMAGES_PATH

# Not SUN397 scene categories -- staging dirs in the release.
EXCLUDED_TOPLEVEL = {"misc", "outliers"}


def scene_from_folder(folder: str) -> str:
    """'b/bus_station/indoor' -> 'bus_station/indoor'."""
    parts = [p for p in folder.strip().strip("/").split("/") if p]
    if parts and len(parts[0]) == 1:      # the a/ b/ c/ alphabetical shard
        parts = parts[1:]
    return "/".join(parts)


def parse_one(xml_path: Path, annotations_root: Path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        try:
            raw = xml_path.read_bytes().translate(_CTRL_TABLE)
            root = ET.fromstring(raw)
        except ET.ParseError as e:
            return None, f"parse_error: {e}"

    folder_el = root.findtext("folder")
    if folder_el:
        folder = folder_el
    else:  # fall back to location on disk
        folder = str(xml_path.parent.relative_to(annotations_root))
    scene = scene_from_folder(folder)
    if not scene:
        return None, "no_scene"
    if scene.split("/")[0] in EXCLUDED_TOPLEVEL:
        return None, "excluded_scene"

    raw_names, kept, n_deleted = [], [], 0
    for obj in root.iter("object"):
        if (obj.findtext("deleted") or "0").strip() == "1":
            n_deleted += 1
            continue
        raw = obj.findtext("name") or ""
        raw_names.append(raw)
        norm = normalise(raw)
        if norm:
            kept.append(norm)

    # types present, order-stable
    objects = list(dict.fromkeys(kept))
    record = {
        "image_id": (root.findtext("filename") or xml_path.stem).rsplit(".", 1)[0],
        "scene": scene,
        "folder": folder.strip().strip("/"),
        "objects": objects,
        "n_object_instances": len(kept),
        "n_deleted": n_deleted,
        "raw_names": raw_names,
    }
    return record, None


def main(annotations_root: Path, out_path: Path, min_objects: int) -> None:
    xml_files = sorted(annotations_root.rglob("*.xml"))
    print(f"Found {len(xml_files)} XML files under {annotations_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped = Counter()
    scenes, objs = Counter(), Counter()
    n_written = 0

    with out_path.open("w") as fh:
        for xml_path in xml_files:
            record, err = parse_one(xml_path, annotations_root)
            if err:
                skipped[err.split(":")[0]] += 1
                continue
            if len(record["objects"]) < min_objects:
                skipped["too_few_objects"] += 1
                continue
            fh.write(json.dumps(record) + "\n")
            n_written += 1
            scenes[record["scene"]] += 1
            objs.update(record["objects"])

    print(f"\nWrote {n_written} image records -> {out_path}")
    print(f"  scene categories : {len(scenes)}")
    print(f"  object types     : {len(objs)}")
    if skipped:
        print("  skipped          : " + ", ".join(f"{k}={v}" for k, v in skipped.most_common()))
    print("\nTop 15 objects:   " + ", ".join(f"{o}({c})" for o, c in objs.most_common(15)))
    print("Top 10 scenes:    " + ", ".join(f"{s}({c})" for s, c in scenes.most_common(10)))
    singletons = sum(1 for c in objs.values() if c == 1)
    print(f"\nObject types seen exactly once: {singletons} "
          f"({singletons / max(len(objs), 1):.1%} of vocabulary)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path,
                    default=ANNOTATIONS_DIR)
    ap.add_argument("--out", type=Path,
                    default=IMAGES_PATH)
    ap.add_argument("--min-objects", type=int, default=2,
                    help="drop images with fewer than this many object types "
                         "(a 1-object image contributes no co-occurrence pair)")
    a = ap.parse_args()
    main(a.annotations, a.out, a.min_objects)
