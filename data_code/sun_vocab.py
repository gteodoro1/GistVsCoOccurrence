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

# Misspellings and spacing variants, applied after singularising. Reviewed by
# hand from close string matches over all types: only clear typos of one
# object are merged; look-alikes that name different objects (brush / bush,
# stable / table, desk chair / deck chair, trowel / towel, soup dish / soap dish)
# are left apart. "bathtube" outnumbers "bathtub" in SUN2012 but is the typo.
SPELLING = {
    # bathroom
    "bathtube": "bathtub", "bathtue": "bathtub", "bath tube": "bathtub",
    "toileto": "toilet", "tootbrush": "toothbrush", "soapdish": "soap dish",
    "shoap dispenser": "soap dispenser", "hairdryer": "hair dryer", "jauzzi": "jacuzzi",
    # furniture and fittings
    "chcair": "chair", "tabale": "table", "cabiet": "cabinet", "cainet": "cabinet",
    "coffe table": "coffee table", "caffee table": "coffee table", "sidetable": "side table",
    "racking chair": "rocking chair", "dek chair": "deck chair", "bookscase": "bookcase",
    "shoowcase": "showcase", "work bench": "workbench", "pilow": "pillow",
    "cushio": "cushion", "blancket": "blanket", "cutain": "curtain", "shuter": "shutter",
    "blinds": "blind", "canoppy": "canopy", "trapestry": "tapestry",
    "billiar table": "billiard table", "table footbal": "table football",
    "doorframe": "door frame", "dloor": "door", "hande door": "handle door",
    "windodw": "window", "wwall": "wall", "wawll": "wall", "stones wall": "stone wall",
    "ceilling": "ceiling", "railin": "railing", "stairscase": "staircase",
    "fire place": "fireplace", "firepace": "fireplace", "chimne": "chimney",
    "plataform": "platform", "outlete": "outlet",
    # lights
    "ceilin lamp": "ceiling lamp", "ceiing lamp": "ceiling lamp",
    "ceiling lam": "ceiling lamp", "ceiling lampo": "ceiling lamp",
    "scocnce": "sconce", "scoce": "sconce", "cadleholder": "candleholder",
    "street light": "streetlight", "streetlihgt": "streetlight", "steetlight": "streetlight",
    # kitchen and table
    "fauecet": "faucet", "regrigerator": "refrigerator", "dishwaser": "dishwasher",
    "diswasher": "dishwasher", "dish washer": "dishwasher",
    "extrator hood": "extractor hood", "estractor hood": "extractor hood",
    "kitcken island": "kitchen island", "kinfe set": "knife set", "spices rack": "spice rack",
    "utensil canister": "utensils canister", "tea pot": "teapot", "casserola": "casserole",
    "flork": "fork", "chees": "cheese", "begetable": "vegetable", "fruits bowl": "fruit bowl",
    "plate mat": "place mat", "pace mat": "place mat", "place mate": "place mat",
    "papar roll": "paper roll", "bottlel rack": "bottle rack", "bottel rack": "bottle rack",
    "washing maching": "washing machine", "washig machine": "washing machine",
    "tumbles dryer": "tumble dryer",
    # objects and devices
    "paiting": "painting", "paintig": "painting", "paintingn": "painting",
    "pictue": "picture", "pictrure": "picture", "telphone": "telephone",
    "telephono": "telephone", "loukspeaker": "loudspeaker", "key board": "keyboard",
    "proyection screen": "projection screen", "projector screen": "projection screen",
    "projection screem": "projection screen", "projection sceen": "projection screen",
    "suitcas": "suitcase", "pursse": "purse", "pourse": "purse", "blackpack": "backpack",
    "brocchure": "brochure", "hashtray": "ashtray", "potterly": "pottery",
    "wasterpaper basket": "wastepaper basket", "watepaper basket": "wastepaper basket",
    "paint brush": "paintbrush", "tap wrech": "tap wrench", "strecher": "stretcher",
    "sliper": "slipper", "trouser": "trousers", "cash regiter": "cash register",
    "arcade machien": "arcade machine", "arcade machie": "arcade machine",
    "sloot machine": "slot machine", "intrument panel": "instrument panel",
    "score board": "scoreboard", "basket ball hoop": "basketball hoop",
    "splinkler": "sprinkler", "tolley": "trolley", "pick-up": "pickup",
    "trafic cone": "traffic cone", "sattelite dish": "satellite dish",
    "aricraft carrier": "aircraft carrier", "wieghbridge": "weighbridge",
    "waterwheel": "water wheel",
    # people and nature
    "perso": "person", "perso sitting": "person sitting", "perso sittig": "person sitting",
    "perso standing": "person standing", "people sitting": "person sitting",
    "people standing": "person standing", "people walking": "person walking", "plalnt": "plant", "polant": "plant",
    "planta": "plant", "plant por": "plant pot", "panter": "planter", "flowr": "flower",
    "florwer": "flower", "dread flower": "dried flower", "plam tree": "palm tree",
    "trees trunk": "tree trunk", "moutain": "mountain", "sand beacoh": "sand beach",
    "see water": "sea water", "swimning pool": "swimming pool",
    "bowlings alley": "bowling alley",
}

_DIGIT_SUFFIX = re.compile(r"\s+\d+$")
_NON_NAME = re.compile(r"[^a-z0-9 \-']")


def singularise(name: str) -> str:
    if name in KEEP_PLURAL:
        return name
    # "chest of drawers": the head noun is before "of", the rest stays as typed
    if " of " in name:
        head, _, tail = name.partition(" of ")
        return f"{singularise(head)} of {tail}"
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
    if not do_singularise:
        return name
    name = singularise(name)
    return SPELLING.get(name, name)
