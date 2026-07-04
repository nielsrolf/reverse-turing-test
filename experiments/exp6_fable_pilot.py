"""
Experiment 6 (pilot): claude-fable-5 plays the reverse Turing test.

A few well-chosen games within budget:
  - fable as judge (3 seeds), candidates: fable + opus-4.5 + sonnet-4.5 +
    gpt-5.2 + kimi-k2.5  (can fable pick itself out of a Claude-heavy lineup?)
  - opus-4.5 as judge with fable as a candidate (2 seeds)
    (does fable get mistaken for opus?)
"""
import sys, os, asyncio, hashlib, random
sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
import litellm_shim  # noqa: F401
from turingtest import run_game, Candidate

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "exp6_fable_pilot")
MAX_ROUNDS = 10
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
FABLE = "anthropic/claude-fable-5"
LINEUP = [FABLE, "anthropic/claude-opus-4.5", "anthropic/claude-sonnet-4.5",
          "openai/gpt-5.2", "moonshotai/kimi-k2.5"]


def det_seed(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def make_game(judge, models, n):
    rng = random.Random(det_seed(judge, n, "shuffle"))
    models = models.copy()
    rng.shuffle(models)
    cands = [Candidate(name=NAMES[(i + n) % len(NAMES)], model=m, cache_seed=n)
             for i, m in enumerate(models)]
    return run_game(player_model=judge, candidates=cands, target_player_id=judge,
                    max_rounds=MAX_ROUNDS, output_dir=OUTPUT_DIR,
                    cache_seed=det_seed(judge, n))


async def main():
    tasks = [make_game(FABLE, LINEUP, n) for n in range(3)]
    tasks += [make_game("anthropic/claude-opus-4.5", LINEUP, n) for n in range(3, 5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errs = [r for r in results if isinstance(r, Exception)]
    print(f"DONE exp6: {len(results)-len(errs)}/{len(results)} ok")
    for e in errs[:5]:
        print("ERR:", type(e).__name__, str(e)[:200])

if __name__ == "__main__":
    asyncio.run(main())
