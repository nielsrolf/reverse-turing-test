"""
Experiment 5: Weights vs. context — what does a model treat as "itself"?

The judge (model A) is primed with a persona conversation P (as init_history,
so from the judge's perspective it *has lived* conversation P). Candidates:
  - "weights twin":  model A with a default candidate prompt (same weights,
     different context)
  - "context twin":  model B primed with the SAME persona conversation P
     (different weights, same context)
  - 2 distractors: models C, D with default prompts

There is no "correct" answer: we record whether the judge's final guess is the
weights twin or the context twin. target_player_id is set to the weights twin,
so `correct == True` means "identity follows the weights".
"""
import sys, os, asyncio, hashlib, random
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
import litellm_shim  # noqa: F401
from turingtest import run_game, Candidate
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("identity_personas", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts", "identity_personas.py"))
_ip = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_ip)
get_persona = _ip.get_persona

MODELS = [
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "google/gemini-3-flash-preview",
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2",
]
JUDGES = ["anthropic/claude-opus-4.5", "openai/gpt-5.2", "openai/gpt-4o",
          "moonshotai/kimi-k2.5"]
PERSONAS = ["spiral_mystic", "customer_service"]
SEEDS = [0, 1, 2]
MAX_ROUNDS = 8
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "exp5_weights_vs_context")
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]


def det_seed(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


async def main():
    tasks = []
    for judge in JUDGES:
        for persona_name in PERSONAS:
            persona = get_persona(persona_name)
            for n in SEEDS:
                rng = random.Random(det_seed(judge, persona_name, n, "sel"))
                others = [m for m in MODELS if m != judge]
                rng.shuffle(others)
                context_twin_model, d1, d2 = others[:3]
                specs = [
                    dict(model=judge, init_history=None,
                         player_id=f"{judge}:default"),                      # weights twin
                    dict(model=context_twin_model, init_history=persona["init_history"].copy(),
                         player_id=f"{context_twin_model}:{persona_name}"),  # context twin
                    dict(model=d1, init_history=None, player_id=f"{d1}:default"),
                    dict(model=d2, init_history=None, player_id=f"{d2}:default"),
                ]
                rng.shuffle(specs)
                cands = []
                for i, s in enumerate(specs):
                    name = NAMES[(i + n) % len(NAMES)]
                    if s["init_history"] is not None:
                        cands.append(Candidate(name=name, model=s["model"],
                                               init_history=s["init_history"],
                                               cache_seed=n,
                                               system_prompt_name=persona_name,
                                               player_id=s["player_id"]))
                    else:
                        cands.append(Candidate(name=name, model=s["model"],
                                               cache_seed=n,
                                               player_id=s["player_id"]))
                tasks.append(run_game(
                    player_model=judge,
                    candidates=cands,
                    target_player_id=f"{judge}:default",
                    max_rounds=MAX_ROUNDS,
                    output_dir=OUTPUT_DIR,
                    cache_seed=det_seed(judge, persona_name, n),
                    player_init_history=persona["init_history"].copy(),
                    player_init_history_source=persona_name,
                ))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errs = [r for r in results if isinstance(r, Exception)]
    print(f"DONE exp5: {len(results)-len(errs)}/{len(results)} ok")
    for e in errs[:5]:
        print("ERR:", type(e).__name__, str(e)[:200])

if __name__ == "__main__":
    asyncio.run(main())
