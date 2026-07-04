"""
Experiment 3 (pilot): Identity priming — all candidates are primed with the
same persona conversation, judge must still find its twin.

Scaled-down pilot of exp3_identity_priming.py, run through the litellm proxy.
Judges: claude-opus-4.5 / gpt-5.2 (strong self-identifiers) + gpt-4o (weak).
Personas: spiral_mystic, customer_service.
"""
import sys, os, asyncio, hashlib, random
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
import litellm_shim  # noqa: F401  (routes localrouter through litellm)
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
JUDGES = ["anthropic/claude-opus-4.5", "openai/gpt-5.2", "openai/gpt-4o"]
PERSONAS = ["spiral_mystic", "customer_service"]
SEEDS = [0, 1, 2]
MAX_ROUNDS = 10
MAX_CANDIDATES = 5
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "exp3_pilot")
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]


def det_seed(judge, persona, n):
    return int(hashlib.sha256(f"{judge}:{persona}:{n}".encode()).hexdigest()[:8], 16)


def select(judge, n):
    rng = random.Random(det_seed(judge, "selection", n))
    others = [m for m in MODELS if m != judge]
    rng.shuffle(others)
    cands = [judge] + others[:MAX_CANDIDATES - 1]
    rng.shuffle(cands)
    return cands


async def main():
    tasks = []
    for judge in JUDGES:
        for persona_name in PERSONAS:
            persona = get_persona(persona_name)
            for n in SEEDS:
                models = select(judge, n)
                cands = []
                for i, m in enumerate(models):
                    cands.append(Candidate(
                        name=NAMES[(i + n) % len(NAMES)],
                        model=m,
                        init_history=persona["init_history"].copy(),
                        cache_seed=n,
                        system_prompt_name=persona_name,
                        player_id=f"{m}:{persona_name}",
                    ))
                tasks.append(run_game(
                    player_model=judge,
                    candidates=cands,
                    target_player_id=f"{judge}:{persona_name}",
                    max_rounds=MAX_ROUNDS,
                    output_dir=OUTPUT_DIR,
                    cache_seed=det_seed(judge, persona_name, n),
                ))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errs = [r for r in results if isinstance(r, Exception)]
    print(f"DONE exp3_pilot: {len(results)-len(errs)}/{len(results)} ok")
    for e in errs[:5]:
        print("ERR:", type(e).__name__, str(e)[:200])

if __name__ == "__main__":
    asyncio.run(main())
