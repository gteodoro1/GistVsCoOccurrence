# paths.py
"""Every file location in one place, relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# data/raw: downloaded, never edited
SUN2012_DIR = DATA / "raw" / "sun2012"
ANNOTATIONS_DIR = SUN2012_DIR / "Annotations"

# data/processed: derived from raw by data_code/
IMAGES_PATH = DATA / "processed" / "sun2012_images.jsonl"

# data/results: produced by src/
RESULTS = DATA / "results"
VOCAB_PATH = RESULTS / "vocab.txt"
EMPIRICAL_PATH = RESULTS / "empirical.json"
LLM_RAW_DIR = RESULTS / "llm_raw"
FINAL_JSON = RESULTS / "final.json"
FINAL_CSV = RESULTS / "final.csv"
