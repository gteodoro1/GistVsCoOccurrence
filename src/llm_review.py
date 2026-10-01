# src/llm_review.py
"""Step 2 of 3: an LLM reviews each empirical top-10 and completes or corrects it.

For every target the model sees the empirical top-10 with its evidence (shared
image count, NPMI, P(anchor | target)), the next 20 runners-up, and the full
vocabulary from step 1. It returns a final list of up to K objects, each tagged
with where it came from:

    kept      an empirical top-10 anchor, retained
    promoted  an empirical runner-up moved into the list
    added     not in the empirical candidates; the model's own knowledge

plus the empirical anchors it removed, each with a category (`why`) and a reason.

How far the model may depart from the data depends on the target's trust level,
set in step 1 from how many images the target appears in (see TRUST_RULES):
    data      frequent target: keep the data; remove only variants, parts and
              artefacts (or a SPARSE anchor), and add nothing of its own
    balanced  correct and complete, with at most MAX_ADDED_BALANCED own objects
    llm       rare target: the statistics rest on few scenes, so the model's
              own knowledge leads and the empirical list is a hint
These are checked in code, not only asked for in the prompt: a reply that adds
too much, removes for a reason its level does not allow, or drops an empirical
anchor without listing it under "removed" is sent back like any other error.

The model is Qwen3-30B-A3B, quantised to 4 bits so it runs locally on Apple
Silicon through MLX (about 17 GB of memory). Decoding is greedy, so a rerun with
the same weights gives the same lists. Output is not schema-constrained, so the
reply is parsed and checked in code: malformed JSON, names outside the
vocabulary and duplicates are quoted back to the model and it retries, up to
MAX_ATTEMPTS. The source tag is set in code from the lists, not trusted from the
model. Each target's prompt and every raw reply are stored under llm_raw/, so a
rerun skips targets with a valid result and the exact exchange behind every list is kept.

The first run downloads the weights from Hugging Face.
"""
import argparse
import json
import sys
import time

from src.paths import EMPIRICAL_PATH, LLM_RAW_DIR as RAW_DIR, VOCAB_PATH

MODEL_ID = "mlx-community/Qwen3-30B-A3B-4bit"
K = 10
MAX_ATTEMPTS = 2              # a third attempt fixed nothing in testing
MAX_NEW_TOKENS = 2048          # answer only; raised automatically when thinking is on
MAX_NEW_TOKENS_THINKING = 8192
# Stored with every record; a record from other rules counts as not done, so a
# change here reruns the targets it affects without --redo-llm.
RULES_VERSION = 4

# Categories for removing an empirical anchor.
WHY = ["variant", "part", "artefact", "sparse", "weaker"]
WHY_TEXT = """  variant   a variant, misspelling or synonym of the target
  part      a part of the target
  artefact  an annotation artefact, not a real object in the scene
  sparse    rests on few images or an unrepresentative sample
  weaker    a clearly stronger co-occurring object should take its place"""

MAX_ADDED_BALANCED = 3
TRUST_RULES = {
    "data": {
        "max_added": 0,
        "why": {"variant", "part", "artefact"},   # + "sparse" for a SPARSE anchor
        "text": "Trust level: DATA. This target appears in {df} images, enough for its "
                "statistics to be reliable. Keep every empirical anchor unless it is a "
                "variant, a part of the target or an artefact; an anchor marked SPARSE may "
                "also be removed as sparse. Do not add objects of your own: fill any gap "
                "with runners-up only.",
    },
    "balanced": {
        "max_added": MAX_ADDED_BALANCED,
        "why": set(WHY),
        "text": "Trust level: BALANCED. This target appears in {df} images. Keep the "
                "empirical anchors by default, correct them for any of the removal "
                "categories, and add at most " + str(MAX_ADDED_BALANCED) + " objects of "
                "your own.",
    },
    "llm": {
        "max_added": K,
        "why": set(WHY),
        "text": "Trust level: LLM. This target appears in only {df} images, so its "
                "statistics rest on few scenes and may reflect those scenes rather than "
                "where the object is usually found. Treat the empirical list as a hint: "
                "keep anchors that fit, and replace the rest with runners-up or objects "
                "of your own, based on your knowledge of where this object occurs.",
    },
}

SYSTEM_TEMPLATE = """You are helping build, for a research thesis, a list of the objects that \
co-occur with a target object in real-world scenes: objects a person would expect to see in \
the same scene as the target, and could use to find it.

The starting point for each target is co-occurrence statistics from SUN2012, a dataset of \
15,017 photographs with human object annotations. Those statistics are real evidence, but they \
have known weaknesses:
- Annotators typed labels freely, so the vocabulary contains variants of one object \
("person" / "person sitting" / "person standing", "door" / "door frame").
- Parts of the target are annotated as separate objects ("headboard" for bed).
- Rare targets rest on few images, so their statistics can reflect whichever few scenes \
happened to be photographed rather than where the object usually occurs.

Your task for each target: return the final list of up to {k} objects that most strongly \
co-occur with it in real scenes, ordered from strongest to weakest.

How much you may change the data depends on how many images the target appears in. Each \
target comes with a trust level:
- DATA: a frequent target. The statistics decide; you only clean them.
- BALANCED: you may correct the list and add a few objects of your own.
- LLM: a rare target. Its statistics rest on few scenes, so your own knowledge leads.

Rules:
1. Follow the trust level given for the target.
2. Every empirical top-{k} anchor must appear either in "anchors" or in "removed"; never drop \
one silently. "removed" holds only empirical top-{k} anchors.
3. Each removal gets one of these categories ("why"):
{why}
4. To fill a gap, prefer a runner-up (it has data behind it) over an object of your own.
5. Every object must be written exactly as it appears in the vocabulary below. Never use the \
target itself or one of its variants.
6. Return fewer than {k} only if there are genuinely not {k} reasonable co-occurring objects \
in the vocabulary.

Answer with a single JSON object and nothing else, in this shape:
{{"anchors": [{{"object": "...", "source": "kept" | "promoted" | "added", "reason": "..."}}],
  "removed": [{{"object": "...", "why": "...", "reason": "..."}}],
  "data_quality_note": "..."}}
"source" is "kept" for an object from the empirical top-{k}, "promoted" for a runner-up, and \
"added" for any other vocabulary object. Keep each reason to one short sentence.

Vocabulary ({n_vocab} object types):
{vocab}"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object": {"type": "string"},
                    "source": {"type": "string", "enum": ["kept", "promoted", "added"]},
                    "reason": {"type": "string"},
                },
                "required": ["object", "source", "reason"],
                "additionalProperties": False,
            },
        },
        "removed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "object": {"type": "string"},
                    "why": {"type": "string", "enum": WHY},
                    "reason": {"type": "string"},
                },
                "required": ["object", "why", "reason"],
                "additionalProperties": False,
            },
        },
        "data_quality_note": {"type": "string"},
    },
    "required": ["anchors", "removed", "data_quality_note"],
    "additionalProperties": False,
}


def fmt_rows(rows):
    if not rows:
        return "  (none)"
    return "\n".join(
        f"  {r['rank']:>2}. {r['object']}  (shared images={r['count']}, "
        f"npmi={r['npmi']}, P(obj|target)={r['p_given']}"
        f"{', SPARSE' if r['sparse'] else ''})"
        for r in rows)


def user_prompt(target, info, n_images):
    trust = TRUST_RULES[info["trust"]]["text"].format(df=info["df"])
    return f"""Target: {target}
Appears in {info['df']} of {n_images} images. Empirical status: {info['status']} \
({info['n_candidates']} candidates with >=3 shared images and positive NPMI; SPARSE means \
fewer than 5 shared images).

{trust}

Empirical top-{K} (ranked by NPMI):
{fmt_rows(info['anchors'])}

Runners-up:
{fmt_rows(info['runners_up'])}

Return the final list for "{target}"."""


def validate(result, target, vocab, info):
    """Return a list of problems; empty means the result is usable."""
    empirical_top = {a["object"] for a in info["anchors"]}
    runners = {a["object"] for a in info["runners_up"]}
    sparse = {a["object"] for a in info["anchors"] if a["sparse"]}
    rules = TRUST_RULES[info["trust"]]
    problems = []
    names = [a["object"] for a in result["anchors"]]
    bad = [n for n in names if n not in vocab]
    if bad:
        problems.append(f"not in the vocabulary: {bad}")
    if target in names:
        problems.append("the target itself is listed")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(f"duplicates: {dupes}")
    if len(names) > K:
        problems.append(f"{len(names)} objects listed, the maximum is {K}")
    # The source tag follows from the lists, so set it here rather than spend a
    # retry on bookkeeping; the model's own tag is kept for the record.
    for a in result["anchors"]:
        a.setdefault("model_source", a["source"])
        a["source"] = ("kept" if a["object"] in empirical_top
                       else "promoted" if a["object"] in runners else "added")

    level = info["trust"].upper()
    added = [a["object"] for a in result["anchors"] if a["source"] == "added"]
    if len(added) > rules["max_added"]:
        problems.append(f"trust level {level} allows at most {rules['max_added']} objects "
                        f"that are not empirical anchors or runners-up, you added {added}")
    # Bookkeeping the model gets wrong but that changes no list is fixed here,
    # not retried: removals of non-top-10 objects are set aside, and outside the
    # DATA level an anchor dropped without comment is recorded as 'unstated'.
    result.setdefault("removed_other", [])
    result["removed_other"] += [r for r in result["removed"] if r["object"] not in empirical_top]
    result["removed"] = [r for r in result["removed"] if r["object"] in empirical_top]
    removed = [r["object"] for r in result["removed"]]
    both = sorted(set(removed) & set(names))
    if both:
        problems.append(f"listed both in 'anchors' and in 'removed': {both}; "
                        "keep each object in only one of them")
    silent = sorted(empirical_top - set(names) - set(removed))
    if silent and info["trust"] == "data":
        problems.append(f"these empirical anchors are neither kept nor in 'removed': {silent}")
    elif silent:
        result["removed"] += [{"object": o, "why": "unstated", "reason": ""} for o in silent]
    for r in result["removed"]:
        if r["why"] != "unstated" and r["why"] not in rules["why"] and not (
                r["why"] == "sparse" and r["object"] in sparse):
            problems.append(f"trust level {level} does not allow removing '{r['object']}' "
                            f"as '{r['why']}'; keep it")
    return problems


def salvage(result, target, vocab, info):
    """After the last attempt, drop the entries the model could not fix, so that
    one bad name does not discard the whole list: names outside the vocabulary,
    the target itself, repeats, anything past K, and removals of objects it also
    kept (its "anchors" list is taken as its answer). Returns what was dropped;
    the trimmed result is then validated again."""
    seen, keep, dropped = set(), [], []
    for a in result["anchors"]:
        o = a["object"]
        if o not in vocab or o == target or o in seen or len(keep) == K:
            dropped.append(o)
        else:
            seen.add(o)
            keep.append(a)
    result["anchors"] = keep
    dropped += [r["object"] for r in result["removed"] if r["object"] in seen]
    result["removed"] = [r for r in result["removed"] if r["object"] not in seen]
    return dropped


def check_shape(result):
    """Generation is not schema-constrained, so enforce OUTPUT_SCHEMA here."""
    if not isinstance(result, dict):
        return ["the reply is not a JSON object"]
    problems = [f"missing key '{k}'" for k in OUTPUT_SCHEMA["required"] if k not in result]
    if problems:
        return problems
    if not isinstance(result["anchors"], list) or not isinstance(result["removed"], list):
        return ["'anchors' and 'removed' must be lists"]
    if not isinstance(result["data_quality_note"], str):
        problems.append("'data_quality_note' must be a string")
    sources = OUTPUT_SCHEMA["properties"]["anchors"]["items"]["properties"]["source"]["enum"]
    for a in result["anchors"]:
        if not (isinstance(a, dict) and all(isinstance(a.get(k), str)
                                            for k in ("object", "source", "reason"))):
            problems.append(f"anchor entry needs string object/source/reason: {a}")
        elif a["source"] not in sources:
            problems.append(f"'{a['object']}' has source '{a['source']}', must be one of {sources}")
    for r in result["removed"]:
        if not (isinstance(r, dict) and all(isinstance(r.get(k), str)
                                            for k in ("object", "why", "reason"))):
            problems.append(f"removed entry needs string object/why/reason: {r}")
        elif r["why"] not in WHY:
            problems.append(f"'{r['object']}' has why '{r['why']}', must be one of {WHY}")
    return problems


def parse_reply(text):
    """Pull the JSON object out of a reply that may carry a code fence or stray prose."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[start:end + 1])


class Qwen:
    def __init__(self, model_id, thinking):
        self.model_id = model_id
        self.thinking = thinking
        self.max_new_tokens = MAX_NEW_TOKENS_THINKING if thinking else MAX_NEW_TOKENS
        from mlx_lm import load       # imported here so merge.py can read the rules without MLX
        self.model, self.tok = load(model_id)

    def __call__(self, messages):
        prompt = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=self.thinking)
        from mlx_lm import stream_generate
        text, last = "", None
        for last in stream_generate(self.model, self.tok, prompt,   # greedy by default
                                    max_tokens=self.max_new_tokens):
            text += last.text
        if last.finish_reason == "length":
            raise RuntimeError(f"hit max_new_tokens ({self.max_new_tokens})")
        # Qwen3 puts its reasoning before </think>; only the part after it is the answer
        thinking, _, answer = text.rpartition("</think>")
        return answer.strip(), thinking.replace("<think>", "").strip(), {
            "prompt_tokens": last.prompt_tokens, "new_tokens": last.generation_tokens,
            "tokens_per_s": round(last.generation_tps, 1)}


def refine(llm, target, info, n_images, system, vocab):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_prompt(target, info, n_images)}]
    attempts, result = [], None
    for _ in range(MAX_ATTEMPTS):
        answer, thinking, usage = llm(messages)
        try:
            result = parse_reply(answer)
            problems = check_shape(result)
        except ValueError as e:      # json.JSONDecodeError is a ValueError
            result, problems = None, [f"the reply is not valid JSON ({e})"]
        shape_ok = not problems
        if shape_ok:
            problems = validate(result, target, vocab, info)
        attempts.append({"reply": answer, "thinking": thinking, "parsed": result,
                         "problems": problems, "usage": usage})
        if not problems:
            break
        messages += [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Fix these problems and return the full JSON again: "
                                        + "; ".join(problems)},
        ]
    salvaged = []
    if attempts[-1]["problems"] and shape_ok:
        salvaged = salvage(result, target, vocab, info)
        attempts[-1]["after_salvage"] = validate(result, target, vocab, info)
    valid = not attempts[-1].get("after_salvage", attempts[-1]["problems"])
    return {"target": target, "trust": info["trust"], "rules_version": RULES_VERSION,
            "model": llm.model_id, "thinking": llm.thinking,
            "decoding": "greedy", "user_prompt": messages[1]["content"],
            "valid": valid, "salvaged": salvaged, "final": result,
            "attempts": attempts}


def main(model_id, thinking, only, limit, force, hours=None):
    emp = json.load(EMPIRICAL_PATH.open())
    vocab_list = [v for v in VOCAB_PATH.read_text().splitlines() if v.strip()]
    vocab = set(vocab_list)
    system = SYSTEM_TEMPLATE.format(k=K, why=WHY_TEXT, n_vocab=len(vocab_list),
                                    vocab="\n".join(vocab_list))
    n_images = emp["meta"]["n_images"]

    targets = only or list(emp["targets"])
    unknown = [t for t in targets if t not in emp["targets"]]
    if unknown:
        sys.exit(f"not in the vocabulary: {unknown}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    def finished(t):
        p = RAW_DIR / f"{t.replace(' ', '_').replace('/', '-')}.json"
        if not p.exists():
            return False
        # An invalid record under the current rules counts as done: decoding is
        # greedy, so a rerun would give the same reply. It falls back to its
        # empirical list at merge; --redo-llm reruns it anyway.
        return json.load(p.open()).get("rules_version") == RULES_VERSION
    todo = [t for t in targets if force or not finished(t)]   # missing or old-rules ones run
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print("nothing to do")
        return

    print(f"loading {model_id} (thinking={'on' if thinking else 'off'}) ...")
    llm = Qwen(model_id, thinking)

    # Each target is saved as soon as it is done, so the run can stop at any point
    # (Ctrl+C, the --hours budget, the laptop closing) and the next run resumes.
    start, done = time.monotonic(), 0
    try:
        for i, target in enumerate(todo, 1):
            elapsed = time.monotonic() - start
            if hours is not None and elapsed > hours * 3600:
                print(f"\n--hours {hours} reached, stopping before '{target}'")
                break
            out = RAW_DIR / f"{target.replace(' ', '_').replace('/', '-')}.json"
            try:
                record = refine(llm, target, emp["targets"][target], n_images, system, vocab)
            except RuntimeError as e:
                print(f"  FAILED {target}: {e}", file=sys.stderr)
                continue
            out.write_text(json.dumps(record, indent=1) + "\n")
            done += 1
            per = (time.monotonic() - start) / done
            anchors = record["final"]["anchors"] if record["final"] else []
            print(f"[{i}/{len(todo)}, {per:.0f}s each, ~{per * (len(todo) - i) / 3600:.1f}h left] "
                  f"{target}{'' if record['valid'] else '  (INVALID)'}: "
                  + ", ".join(f"{a['object']}{'' if a['source'] == 'kept' else '*'}"
                              for a in anchors))
    except KeyboardInterrupt:
        print("\nstopped; the target in progress is lost, all finished ones are saved")

    n_done = sum(finished(t) for t in emp["targets"])
    print(f"\n{done} done this run, {n_done}/{len(emp['targets'])} in total"
          + ("" if n_done == len(emp["targets"]) else "; run the same command again to continue"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID,
                    help="MLX model id on Hugging Face, e.g. mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--thinking", action="store_true",
                    help="let Qwen3 reason before answering (slower, often better)")
    ap.add_argument("--only", nargs="+", help="run just these targets")
    ap.add_argument("--limit", type=int, help="stop after this many new targets")
    ap.add_argument("--force", action="store_true", help="redo targets already in llm_raw/")
    ap.add_argument("--hours", type=float, help="stop starting new targets after this many hours")
    a = ap.parse_args()
    main(a.model, a.thinking, a.only, a.limit, a.force, a.hours)
