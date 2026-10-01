# data_code/download_sun2012.py
"""Fetch SUN2012 annotations from the live CSAIL mirror.

The original host (groups.csail.mit.edu/vision/SUN/releases/) is dead -- every
release URL there 404s as of 2026-09. The surviving copy is a directory mirror
under people.csail.mit.edu, which serves an *uncompressed* SUN2012.tar
(4.19 GB, Last-Modified 2014-12-07, supports byte ranges).

Annotations/ is stored entirely before Images/ in that tar, so for co-occurrence
statistics we stream the archive and stop as soon as the first non-Annotations
member appears. That pulls ~200 MB instead of 4.19 GB. Pass --images to keep
going and extract the JPEGs too.
"""
import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://people.csail.mit.edu/aespielberg/SUN2012"
TAR_URL = f"{BASE_URL}/SUN2012.tar"
from src.paths import SUN2012_DIR as DEST_DIR

# tarfile's `filter=` sanitiser only exists from 3.11.4; this box runs 3.9.
EXTRACT_KW = {"filter": "data"} if sys.version_info >= (3, 12) else {}


def stream_extract(dest_dir: Path, want_images: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {TAR_URL}")
    print(f"  -> {dest_dir}  (annotations only: {not want_images})")

    req = urllib.request.Request(TAR_URL, headers={"User-Agent": "thesis-sun2012/1.0"})
    n_xml = 0
    seen_annotations = False

    with urllib.request.urlopen(req) as resp:
        # 'r|' = stream mode: never seeks, so we can abandon the connection early.
        with tarfile.open(fileobj=resp, mode="r|") as tar:
            for member in tar:
                name = member.name
                in_annotations = name.startswith("Annotations/")

                if in_annotations:
                    seen_annotations = True
                elif seen_annotations and not want_images:
                    # Past the annotation block; everything left is image data.
                    print(f"\nReached {name!r} -- stopping before the image payload.")
                    break

                if not member.isfile():
                    continue
                if in_annotations and not name.endswith(".xml"):
                    continue  # .DS_Store and friends

                tar.extract(member, path=dest_dir, **EXTRACT_KW)
                if in_annotations:
                    n_xml += 1
                    if n_xml % 1000 == 0:
                        print(f"  {n_xml} annotation files", flush=True)

    print(f"Done: {n_xml} annotation XML files under {dest_dir / 'Annotations'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", action="store_true",
                    help="also extract Images/ (adds ~4 GB; not needed for co-occurrence)")
    ap.add_argument("--dest", type=Path, default=DEST_DIR)
    args = ap.parse_args()
    try:
        stream_extract(args.dest, args.images)
    except KeyboardInterrupt:
        sys.exit(130)
