# main.py
"""Build, for every SUN2012 object type, the list of objects that co-occur with it.

    python main.py

runs the whole pipeline and ends with data/results/final.json and final.csv. Each stage is skipped when its output already exists, and the LLM
stage resumes where it stopped, so an interrupted run can simply be restarted.

    0  download   SUN2012 annotations (~200 MB)     -> data/raw/sun2012/Annotations/
    1  parse      one record per image              -> data/processed/sun2012_images.jsonl
    2  empirical  top-10 per object from the data   -> data/results/empirical.json
    3  llm        Qwen3 reviews and completes them  -> data/results/llm_raw/
    4  merge      the final lists                   -> data/results/final.{json,csv}

data_code/ holds stages 0-1 (getting the data), src/ holds stages 2-4 (the method).

The LLM stage takes hours. It can be split over several sessions: stop it with
Ctrl+C or give it a budget with --hours 2, and the same command continues later.

Stage 2 is cheap and always reruns, so a change to its settings reaches the
final lists. Use --redo-llm to rerun stage 3 for targets it already finished.
"""
import argparse

from data_code import download_sun2012, parse_annotations
from src import empirical, merge, paths


def stage(n, name):
    print(f"\n=== {n} {name} ===")


def main(a):
    stage(0, "download")
    if paths.ANNOTATIONS_DIR.is_dir():
        print("annotations present, skipping")
    else:
        download_sun2012.stream_extract(paths.SUN2012_DIR, want_images=False)

    stage(1, "parse")
    if paths.IMAGES_PATH.exists():
        print(f"{paths.IMAGES_PATH.name} present, skipping")
    else:
        parse_annotations.main(paths.ANNOTATIONS_DIR, paths.IMAGES_PATH, min_objects=2)

    stage(2, "empirical")
    empirical.main(paths.IMAGES_PATH, paths.EMPIRICAL_PATH, paths.VOCAB_PATH)

    if a.skip_llm:
        print("\n--skip-llm: final lists fall back to the empirical top-10")
    else:
        stage(3, "llm")
        from src import llm_review    # imported here so the other stages run without MLX
        llm_review.main(a.model or llm_review.MODEL_ID, a.thinking,
                        a.only, a.limit, a.redo_llm, a.hours)

    stage(4, "merge")
    merge.main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="MLX model id (default: mlx-community/Qwen3-30B-A3B-4bit)")
    ap.add_argument("--thinking", action="store_true",
                    help="let Qwen3 reason before answering (slower, often better)")
    ap.add_argument("--only", nargs="+", metavar="OBJECT",
                    help="run the LLM stage for just these objects, e.g. --only bed toilet")
    ap.add_argument("--limit", type=int, help="run the LLM stage for at most this many new objects")
    ap.add_argument("--hours", type=float,
                    help="run the LLM stage for about this many hours, then stop; rerun to continue")
    ap.add_argument("--redo-llm", action="store_true",
                    help="rerun the LLM stage for objects it already finished")
    ap.add_argument("--skip-llm", action="store_true", help="build the lists from the data only")
    main(ap.parse_args())
