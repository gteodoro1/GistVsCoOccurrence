# data_code/sun_vocab.py
"""Normalisation of raw SUN2012/LabelMe object names.

LabelMe labels were typed freely by annotators, so the raw ~4,919 categories
contain trailing whitespace, state modifiers ("bus occluded"), disambiguation
digits ("wall 2"), and inconsistent number ("chair" vs "chairs"). Collapsing
these matters for co-occurrence: "chair"/"chairs" split as separate types would
halve the counts that PMI is computed from.

Kept deliberately conservative -- this is *within-dataset* normalisation only.
Mapping onto THINGSplus (for the COOCo target objects) is a separate step.
"""
import re

# State/annotation modifiers that describe the depiction, not the object type.
MODIFIERS = {
    "occluded", "occl", "crop", "cropped", "truncated", "part", "parts",
    "region", "partial", "blurred", "blurry", "small", "far", "reflection",
}

# Words whose plural form is the conventional name, or where stripping the
# trailing -s changes the referent.
KEEP_PLURAL = {
    "glasses", "stairs", "scissors", "shorts", "jeans", "pants", "clothes",
    "binoculars", "headphones", "trousers", "sunglasses", "bleachers",
    "steps", "blinds", "crosswalk stripes",
}

# Irregular plurals the suffix rules below get wrong ("knives" -> "knive").
IRREGULAR = {
    "people": "person", "men": "man", "women": "woman", "children": "child",
    "teeth": "tooth", "feet": "foot", "mice": "mouse", "geese": "goose",
    "knives": "knife", "lives": "life", "wives": "wife", "loaves": "loaf",
    "leaves": "leaf", "shelves": "shelf", "halves": "half", "scarves": "scarf",
    "hooves": "hoof", "calves": "calf", "wolves": "wolf", "cacti": "cactus",
}

_DIGIT_SUFFIX = re.compile(r"\s+\d+$")
_NON_NAME = re.compile(r"[^a-z0-9 \-']")


def singularise(name: str) -> str:
    if name in KEEP_PLURAL:
        return name
    head, _, last = name.rpartition(" ")
    stem = _singularise_word(last)
    return f"{head} {stem}".strip() if head else stem


def _singularise_word(w: str) -> str:
    if w in IRREGULAR:
        return IRREGULAR[w]
    if len(w) < 4 or w.endswith(("ss", "us", "is", "ics")):
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("es") and w[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def normalise(raw: str, do_singularise: bool = True) -> str:
    """Raw <name> text -> canonical object type. Returns '' if nothing is left."""
    name = raw.lower().replace("_", " ").replace("/", " ")
    name = _NON_NAME.sub(" ", name)
    name = " ".join(name.split())          # collapse whitespace, strip
    name = _DIGIT_SUFFIX.sub("", name)

    # Strip trailing modifiers, repeatedly ("car occluded crop").
    tokens = name.split()
    while len(tokens) > 1 and tokens[-1] in MODIFIERS:
        tokens.pop()
    name = " ".join(tokens)

    if not name or name in MODIFIERS:
        return ""
    return singularise(name) if do_singularise else name
