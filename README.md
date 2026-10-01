# Co-occurrence lists (Step 2a)

For every object type in SUN2012 with at least 10 images (682 types), this builds
a list of up to 10 objects that co-occur with it in real scenes, to be used later
to locate the object. The list combines co-occurrence statistics with a review by
an LLM (Qwen3).

## Run

```bash
/opt/homebrew/bin/python3.13 -m venv .venv      # once, from the repo root
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

The result is `data/results/final.json` and `final.csv` (one row per
target–anchor pair). Each stage is skipped when its output exists. The LLM stage
resumes where it stopped, so an interrupted run can be restarted with the same
command.

Useful flags:
- `--only bed toilet`: run the LLM stage for a few objects first.
- `--skip-llm`: build the lists from the data only.
- `--thinking`: turn on Qwen3's reasoning (slower, often better).
- `--redo-llm`: rerun the LLM stage for objects it already finished.
- `--model`: use another MLX model, e.g. `mlx-community/Qwen3-8B-4bit`.

## Layout

```
main.py          runs everything
data_code/       getting the data: download, parse, label normalisation
src/             the method: empirical, llm_review, merge; paths.py holds every path
data/raw/        downloaded SUN2012 annotations (not in git)
data/processed/  parsed image records (not in git)
data/results/    vocab, empirical lists, raw LLM replies, final lists
writing/         thesis text (local only, not in git)
```

## Pipeline

| Stage | Script | Output (under `data/`) |
|---|---|---|
| 0 download | `data_code/download_sun2012.py` | `raw/sun2012/Annotations/` (~200 MB of the 4.2 GB release) |
| 1 parse | `data_code/parse_annotations.py`, `sun_vocab.py` | `processed/sun2012_images.jsonl` |
| 2 empirical | `src/empirical.py` | `results/empirical.json`, `vocab.txt` |
| 3 llm | `src/llm_review.py` | `results/llm_raw/<object>.json` |
| 4 merge | `src/merge.py` | `results/final.json`, `final.csv` |

A single stage can be run on its own from the repo root, e.g.
`.venv/bin/python -m src.merge`.

## Method

**Parse.** LabelMe XML from SUN2012: 15,017 images with at least two object types.
Deleted objects are dropped. Labels are normalised: modifiers ("bus occluded"),
indices ("wall 2") and plurals are collapsed. Types are de-duplicated per image.

**Empirical.** The vocabulary is every type in at least 10 images. A partner is a
candidate if it shares at least 3 images with the target and has positive NPMI.
Candidates are ranked by NPMI. The top 10 become the anchors and the next 20 are
kept as runners-up. A target is `ok` when its 10 anchors each share at least 5
images, `thin` otherwise, and `none` when it has no candidates.

**LLM review.** The model is Qwen3-30B-A3B, 4-bit, run locally through MLX, with
greedy decoding so the run is reproducible. For each target it sees the anchors
with their evidence, the runners-up and the vocabulary, and returns the final
list. Its instructions:

- Keep the data by default.
- Remove label variants, parts of the target, artefacts and anchors that come from
  an unrepresentative sample.
- Prefer runners-up over its own additions.

The code checks the reply. Bad JSON, names outside the vocabulary, duplicates and
the target itself are sent back to the model, which retries (up to 3 attempts).

**Merge.** Each anchor is tagged by where it came from:

| Tag | Meaning |
|---|---|
| `kept` | an empirical anchor the LLM retained |
| `promoted` | a runner-up the LLM moved into the list |
| `added` | from the LLM's own knowledge |
| `filled` | an empirical object used to top up a short LLM list |
| `empirical` | the target had no valid LLM result |

A simpler `decided_by` column groups these: `empirical` (kept, filled, empirical) when the data put the object in the list, `llm` (promoted, added) when the LLM's decision did.

Every anchor carries its SUN2012 evidence (`count`, `npmi`, `p_given`), so an
`added` object with 0 shared images is visible as such.

## Known limits

- Anchors are restricted to the 682-type vocabulary.
- `added` objects come from text knowledge, not observation. Filter on `source`
  or `count` where a downstream step needs data-backed anchors.
- 4-bit quantisation costs Qwen a little accuracy. A larger machine can run a
  bigger model through `--model`.
