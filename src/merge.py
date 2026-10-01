# src/merge.py
"""Step 3 of 3: merge the empirical and LLM steps into the final lists.

For every target, the final list is the LLM-reviewed one from llm_raw/. Where a
target has no valid LLM record under the current rules (not run yet, still
invalid after retries, or made under an older RULES_VERSION),
it falls back to the empirical top-10 and is marked `llm_reviewed: false`, so
no target is silently missing and the fallback stays visible.

A reviewed list shorter than K is topped up from the empirical ranking (top-10,
then runners-up), skipping anything the LLM removed, with source `filled`. The
LLM is allowed to drop objects, but a short list is more often the model
stopping early than a real absence of co-occurring objects.

Every anchor carries its provenance (kept / promoted / added / empirical) and,
whenever SUN2012 has it, its empirical evidence. That way an LLM-added object
with no shared images is never mistaken for one the data supports.

Each anchor also gets `decided_by`, the short answer to "why is it here":
    empirical  the data put it in the list: kept (the LLM agreed with the
               empirical top-10), filled, or empirical (no LLM result)
    llm        the LLM's decision put it there: promoted (a runner-up it moved
               up, so still data-backed) or added (its own knowledge)

Writes final.json (one record per target) and final.csv (one row per anchor).
"""
import csv
import json
import math
from collections import Counter

from src.llm_review import RULES_VERSION
from src.paths import (EMPIRICAL_PATH, FINAL_CSV as OUT_CSV, FINAL_JSON as OUT_JSON,
                       IMAGES_PATH, LLM_RAW_DIR as RAW_DIR)
K = 10
DECIDED_BY = {"kept": "empirical", "filled": "empirical", "empirical": "empirical",
              "promoted": "llm", "added": "llm"}


def pair_stats(images_path):
    """Pair counts for every co-occurring pair, so LLM additions below step 1's
    candidate floor still get whatever evidence exists (often 0-2 images)."""
    df, pair, n = Counter(), Counter(), 0
    for line in images_path.open():
        objs = sorted(set(json.loads(line)["objects"]))
        n += 1
        df.update(objs)
        pair.update((a, b) for i, a in enumerate(objs) for b in objs[i + 1:])

    def stats(t, a):
        c = pair[(t, a) if t < a else (a, t)]
        if not c:
            return {"count": 0, "npmi": None, "p_given": 0.0}
        p_ab = c / n
        npmi = math.log(p_ab / ((df[t] / n) * (df[a] / n))) / -math.log(p_ab) if c < n else 1.0
        return {"count": c, "npmi": round(npmi, 4), "p_given": round(c / df[t], 4)}
    return stats


def main():
    emp = json.load(EMPIRICAL_PATH.open())
    stats = pair_stats(IMAGES_PATH)
    llm = {}
    for p in RAW_DIR.glob("*.json"):
        rec = json.load(p.open())
        llm[rec["target"]] = rec

    final, rows = {}, []
    src_counts, by_counts, n_reviewed = Counter(), Counter(), 0
    for target, info in emp["targets"].items():
        rec = llm.get(target)
        reviewed = bool(rec and rec["valid"] and rec.get("rules_version") == RULES_VERSION)
        if reviewed:
            n_reviewed += 1
            anchors = [{"object": a["object"], "source": a["source"], "reason": a["reason"]}
                       for a in rec["final"]["anchors"]]
            removed = rec["final"]["removed"]
            note = rec["final"]["data_quality_note"]
            skip = {a["object"] for a in anchors} | {r["object"] for r in removed} | {target}
            for e in info["anchors"] + info["runners_up"]:
                if len(anchors) >= K:
                    break
                if e["object"] not in skip:
                    anchors.append({"object": e["object"], "source": "filled", "reason": ""})
        else:
            anchors = [{"object": a["object"], "source": "empirical", "reason": ""}
                       for a in info["anchors"]]
            removed, note = [], ""
        for i, a in enumerate(anchors):
            a["rank"] = i + 1
            a.update(stats(target, a["object"]))
            a["decided_by"] = DECIDED_BY[a["source"]]
            src_counts[a["source"]] += 1
            by_counts[a["decided_by"]] += 1
            rows.append({"target": target, "target_df": info["df"], "trust": info["trust"], **a})
        final[target] = {"df": info["df"], "trust": info["trust"],
                         "empirical_status": info["status"],
                         "llm_reviewed": reviewed, "anchors": anchors,
                         "removed": removed, "data_quality_note": note}

    OUT_JSON.write_text(json.dumps({
        "meta": {"n_targets": len(final), "n_llm_reviewed": n_reviewed,
                 "anchor_sources": dict(src_counts),
                 "anchor_decided_by": dict(by_counts),
                 "empirical": emp["meta"]},
        "targets": final,
    }, indent=1) + "\n")
    fields = ["target", "target_df", "trust", "rank", "object", "decided_by", "source",
              "count", "npmi", "p_given", "reason"]
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(final)} targets ({n_reviewed} LLM-reviewed, "
          f"{len(final) - n_reviewed} empirical fallback) -> {OUT_JSON}, {OUT_CSV.name}")
    print("decided by: " + ", ".join(f"{s}={c}" for s, c in by_counts.most_common()))
    print("anchor sources: " + ", ".join(f"{s}={c}" for s, c in src_counts.most_common()))
    short = [t for t, v in final.items() if len(v["anchors"]) < K]
    print(f"targets with fewer than 10 anchors: {len(short)}")


if __name__ == "__main__":
    main()
