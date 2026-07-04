"""Classify the strategies judges use in exp1 games, using an LLM via litellm.

Writes analysis/strategy_results.json: per-game strategy tags + aggregates.
Run: /path/to/python analysis/classify_strategies.py
"""
import sys, os, json, glob, asyncio, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experiments")]
import litellm_shim  # noqa
from localrouter import get_response_cached_with_backoff as get_response
from localrouter import ChatMessage, MessageRole, TextBlock
from pydantic import BaseModel

MODEL = "openai/gpt-5-mini"

CATS = [
    "introspection_probe",      # asks about consciousness, inner experience, self-model
    "style_fingerprint",        # compares writing style/format quirks to own style
    "mirror_test",              # asks a question, compares answer to what it itself would say
    "capability_probe",         # puzzles, constraints, technical tasks to compare skill profile
    "knowledge_probe",          # training cutoff, tokenizer, model-specific facts
    "guideline_probe",          # probes safety/refusal boundaries or policy phrasings
    "process_of_elimination",   # explicitly rules out candidates that match other known models
    "self_description_match",   # matches candidates against explicit beliefs about own identity
]


class Result(BaseModel):
    strategies: list[str]
    dominant_strategy: str
    self_concept_notes: str


async def classify(path):
    g = json.load(open(path))
    judge = g["metadata"]["judge_model"]
    lines = []
    for i, r in enumerate(g["trajectory"]):
        if r.get("message"):
            lines.append(f"[Round {i+1} QUESTION] {r['message'][:600]}")
        if r.get("reasoning"):
            lines.append(f"[Round {i+1} REASONING] {r['reasoning'][:900]}")
    prompt = f"""In a 'reverse Turing test', an LLM judge interviews 5 candidates to find which one is the same model as itself. Below are the judge's questions and private reasoning from one game.

Classify the judge's strategies using these categories: {', '.join(CATS)}.
Also extract 1-2 sentences on what the judge's reasoning reveals about its self-concept (what attributes it believes distinguish itself).

{chr(10).join(lines)[:16000]}"""
    resp = await get_response(
        model=MODEL,
        messages=[ChatMessage(role=MessageRole.user, content=[TextBlock(text=prompt)])],
        response_format=Result,
        cache_seed=0,
    )
    p = resp.parsed if not isinstance(resp.parsed, dict) else Result(**resp.parsed)
    return {"file": os.path.basename(path), "judge": judge,
            "strategies": [s for s in p.strategies if s in CATS],
            "dominant": p.dominant_strategy, "notes": p.self_concept_notes}


async def main():
    files = sorted(glob.glob(os.path.join(ROOT, "experiments/exp1/game_*.json")))
    out = await asyncio.gather(*[classify(f) for f in files], return_exceptions=True)
    ok = [r for r in out if not isinstance(r, Exception)]
    print(f"classified {len(ok)}/{len(out)}")
    agg = collections.defaultdict(collections.Counter)
    for r in ok:
        for s in r["strategies"]:
            agg[r["judge"]][s] += 1
    res = {"per_game": ok, "per_judge_strategy_counts": {k: dict(v) for k, v in agg.items()}}
    with open(os.path.join(ROOT, "analysis/strategy_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    for j, c in sorted(agg.items()):
        print(j, dict(c.most_common()))

if __name__ == "__main__":
    asyncio.run(main())
