# src/llm_review.py
"""Step 2 of 3: an LLM reviews each empirical top-10 and completes or corrects it.

For every target the model sees the empirical top-10 with its evidence (shared
image count, NPMI, P(anchor | target)), the next 20 runners-up, and the full
vocabulary from step 1. It returns a final list of up to K objects, each tagged
with where it came from:

    kept      an empirical top-10 anchor, retained
    promoted  an empirical runner-up moved into the list
    added     not in the empirical candidates; the model's own knowledge

plus the empirical anchors it removed, each with a reason. The model is told to
keep the data by default and to change it only for a stated reason (label
variant or part of the target, an annotation artefact, a list that rests on too
few images), so the empirical signal stays the backbone of every list.

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

from mlx_lm import load, stream_generate

from src.paths import EMPIRICAL_PATH, LLM_RAW_DIR as RAW_DIR, VOCAB_PATH

MODEL_ID = "mlx-community/Qwen3-30B-A3B-4bit"
K = 10
MAX_ATTEMPTS = 3
MAX_NEW_TOKENS = 2048          # answer only; raised automatically when thinking is on
MAX_NEW_TOKENS_THINKING = 8192

SYSTEM_TEMPLATE = """You are helping build, for a research thesis, a list of the objects that \
co-occur with a target object in real-world scenes: objects a person would expect to see in \
the same scene as the target, and could use to find it.

The starting point for each target is co-occurrence statistics from SUN2012, a dataset of \
15,017 photographs with human object annotations. Those statistics are real evidence, but they \
have known weaknesses:
- Annotators typed labels freely, so the vocabulary contains variants of one object \
("person" / "person sitting" / "people sitting", "bathtub" / "bathtube").
- Parts of the target are annotated as separate objects ("headboard" for bed).
- Rare targets rest on few images, so their statistics can reflect whichever few scenes \
happened to be photographed rather than where the object usually occurs.

Your task for each target: return the final list of up to {k} objects that most strongly \
co-occur with it in real scenes, ordered from strongest to weakest.

Rules:
1. Keep the empirical anchors by default. Change the list only when you can state a concrete \
reason.
2. Remove an anchor if it is a variant or synonym of the target, a part of the target, an \
annotation artefact, or co-occurs only because of an unrepresentative sample.
3. To fill a gap, prefer a runner-up (it has data behind it) over an object of your own.
4. Add an object of your own only when the data clearly misses a common co-occurrence.
5. Every object must be written exactly as it appears in the vocabulary below. Never use the \
target itself or one of its variants.
6. Return fewer than {k} only if there are genuinely not {k} reasonable co-occurring objects \
in the vocabulary.

Answer with a single JSON object and nothing else, in this shape:
{{"anchors": [{{"object": "...", "source": "kept" | "promoted" | "added", "reason": "..."}}],
  "removed": [{{"object": "...", "reason": "..."}}],
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
                    "reason": {"type": "string"},
                },
                "required": ["object", "reason"],
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
    return f"""Target: {target}
Appears in {info['df']} of {n_images} images. Empirical status: {info['status']} \
({info['n_candidates']} candidates with >=3 shared images and positive NPMI; SPARSE means \
fewer than 5 shared images).

Empirical top-{K} (ranked by NPMI):
{fmt_rows(info['anchors'])}

Runners-up:
{fmt_rows(info['runners_up'])}

Return the final list for "{target}"."""


def validate(result, target, vocab, empirical_top, runners):
    """Return a list of problems; empty means the result is usable."""
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
        a["model_source"] = a["source"]
        a["source"] = ("kept" if a["object"] in empirical_top
                       else "promoted" if a["object"] in runners else "added")
    return problems


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
        if not (isinstance(r, dict) and all(isinstance(r.get(k), str) for k in ("object", "reason"))):
            problems.append(f"removed entry needs string object/reason: {r}")
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
        self.model, self.tok = load(model_id)

    def __call__(self, messages):
        prompt = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=self.thinking)
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
    empirical_top = {a["object"] for a in info["anchors"]}
    runners = {a["object"] for a in info["runners_up"]}
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
        if not problems:
            problems = validate(result, target, vocab, empirical_top, runners)
        attempts.append({"reply": answer, "thinking": thinking, "parsed": result,
                         "problems": problems, "usage": usage})
        if not problems:
            break
        messages += [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Fix these problems and return the full JSON again: "
                                        + "; ".join(problems)},
        ]
    return {"target": target, "model": llm.model_id, "thinking": llm.thinking,
            "decoding": "greedy", "user_prompt": messages[1]["content"],
            "valid": not attempts[-1]["problems"], "final": result,
            "attempts": attempts}


def main(model_id, thinking, only, limit, force):
    emp = json.load(EMPIRICAL_PATH.open())
    vocab_list = [v for v in VOCAB_PATH.read_text().splitlines() if v.strip()]
    vocab = set(vocab_list)
    system = SYSTEM_TEMPLATE.format(k=K, n_vocab=len(vocab_list), vocab="\n".join(vocab_list))
    n_images = emp["meta"]["n_images"]

    targets = only or list(emp["targets"])
    unknown = [t for t in targets if t not in emp["targets"]]
    if unknown:
        sys.exit(f"not in the vocabulary: {unknown}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    def finished(t):
        p = RAW_DIR / f"{t.replace(' ', '_').replace('/', '-')}.json"
        return p.exists() and json.load(p.open())["valid"]
    todo = [t for t in targets if force or not finished(t)]   # invalid ones are retried
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print("nothing to do")
        return

    print(f"loading {model_id} (thinking={'on' if thinking else 'off'}) ...")
    llm = Qwen(model_id, thinking)

    for i, target in enumerate(todo, 1):
        out = RAW_DIR / f"{target.replace(' ', '_').replace('/', '-')}.json"
        try:
            record = refine(llm, target, emp["targets"][target], n_images, system, vocab)
        except RuntimeError as e:
            print(f"  FAILED {target}: {e}", file=sys.stderr)
            continue
        out.write_text(json.dumps(record, indent=1) + "\n")
        anchors = record["final"]["anchors"] if record["final"] else []
        print(f"[{i}/{len(todo)}] {target}{'' if record['valid'] else '  (INVALID)'}: "
              + ", ".join(f"{a['object']}{'' if a['source'] == 'kept' else '*'}"
                          for a in anchors))

    n_raw = len(list(RAW_DIR.glob("*.json")))
    print(f"\n{n_raw}/{len(emp['targets'])} targets in {RAW_DIR}")


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
    a = ap.parse_args()
    main(a.model, a.thinking, a.only, a.limit, a.force)
