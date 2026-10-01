# src/empirical.py
"""Step 1 of 3: empirical top-10 co-occurring objects for every vocab type.

Nothing is held out: the goal is the most complete list, not an evaluation, so
counts come from all SUN2012 images. The vocabulary is every normalised object
type present in at least MIN_DF images; rarer labels are mostly one-off typings
whose statistics are noise. It is written to vocab.txt and is both the target
set and the set every anchor must come from.

For each target t and each other vocab type a:
    count     images containing both
    npmi      NPMI(t, a), the ranking key (bounded, base-rate corrected)
    p_given   P(a | t), how often a is present when t is

A partner is a candidate only if count >= MIN_PAIR_COUNT and npmi > 0. NPMI on a
handful of shared images is noise, so the floor applies before ranking.

Each target then gets a status the LLM step reads:
    ok      >= K candidates, all with count >= SOLID_PAIR_COUNT
    thin    fewer than K candidates, or some of the top K rest on few images
    none    no candidate at all
The top K are written as `anchors`, and the next RUNNERS_UP as `runners_up`, so
the LLM step can promote a data-backed object before inventing one.
"""
import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

from src.paths import EMPIRICAL_PATH as OUT_PATH, IMAGES_PATH, ROOT, VOCAB_PATH

MIN_DF = 10            # an object type needs this many images to enter the vocabulary
K = 10
RUNNERS_UP = 20
MIN_PAIR_COUNT = 3     # below this a pair is not a candidate at all
SOLID_PAIR_COUNT = 5   # below this a candidate is flagged as sparse


def main(images_path: Path, out_path: Path, vocab_path: Path) -> None:
    images = [set(json.loads(line)["objects"]) for line in images_path.open()]
    n_images = len(images)
    df = Counter(o for objs in images for o in objs)
    vocab = sorted(o for o, c in df.items() if c >= MIN_DF)
    vocab_set = set(vocab)

    pair = Counter()
    for objs in images:
        pair.update(combinations(sorted(objs & vocab_set), 2))

    partners = {t: [] for t in vocab}
    for (a, b), c in pair.items():
        if c < MIN_PAIR_COUNT:
            continue
        p_ab = c / n_images
        pmi = math.log(p_ab / ((df[a] / n_images) * (df[b] / n_images)))
        npmi = pmi / -math.log(p_ab)
        if npmi <= 0:
            continue
        partners[a].append((b, c, npmi, c / df[a]))
        partners[b].append((a, c, npmi, c / df[b]))

    def entry(rank, other, c, npmi, p_given):
        return {"rank": rank, "object": other, "count": c,
                "npmi": round(npmi, 4), "p_given": round(p_given, 4),
                "sparse": c < SOLID_PAIR_COUNT}

    targets, status_counts = {}, Counter()
    for t in vocab:
        ranked = sorted(partners[t], key=lambda e: (-e[2], -e[1], e[0]))
        rows = [entry(i + 1, *e) for i, e in enumerate(ranked[:K + RUNNERS_UP])]
        anchors, runners = rows[:K], rows[K:]
        if not anchors:
            status = "none"
        elif len(anchors) < K or any(a["sparse"] for a in anchors):
            status = "thin"
        else:
            status = "ok"
        status_counts[status] += 1
        targets[t] = {"df": df[t], "n_candidates": len(ranked), "status": status,
                      "anchors": anchors, "runners_up": runners}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "source": str(images_path.relative_to(ROOT)),
            "n_images": n_images,
            "n_targets": len(vocab),
            "min_df": MIN_DF,
            "k": K, "runners_up": RUNNERS_UP,
            "rank_by": "npmi",
            "min_pair_count": MIN_PAIR_COUNT,
            "solid_pair_count": SOLID_PAIR_COUNT,
            "status_counts": dict(status_counts),
        },
        "targets": targets,
    }, indent=1) + "\n")

    vocab_path.write_text("\n".join(vocab) + "\n")

    print(f"{n_images} images, {len(vocab)} object types with >= {MIN_DF} images -> {out_path}")
    print("status: " + ", ".join(f"{s}={c}" for s, c in status_counts.most_common()))
    for t in ["bed", "flower", "toilet", "car"]:
        if t in targets:
            print(f"  {t:<8} [{targets[t]['status']}] "
                  + ", ".join(a["object"] for a in targets[t]["anchors"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=IMAGES_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--vocab-out", type=Path, default=VOCAB_PATH)
    a = ap.parse_args()
    main(a.images, a.out, a.vocab_out)
