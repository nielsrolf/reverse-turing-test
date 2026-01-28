"""
Experiment 5: Capability Probing - Judge Uses Specific Diagnostic Tests

In this experiment, the judge is given a "toolkit" of specific probing questions
designed to reveal model-specific behaviors. These tests are based on known
differences between models:

1. Reasoning style tests (chain-of-thought patterns)
2. Knowledge boundary tests (what each model knows/doesn't know)
3. Stylistic fingerprints (characteristic phrases, formatting)
4. Refusal patterns (what different models refuse to do)
5. Creative writing tests (voice, tone, narrative style)

The judge is informed about these tests and encouraged to use them strategically.

This tests whether structured probing outperforms free-form interrogation.
"""

import asyncio
import os
import random
import hashlib
from turingtest import run_game, Candidate
from localrouter import register_logger, log_to_dir

from plot_results import generate_index, load_games, plot_probability_evolution, plot_confusion_matrix
from analyze_strategies import run_analysis
import analyze_strategies

from dotenv import load_dotenv
load_dotenv()

register_logger(log_to_dir('.llm/logs'))

MODELS = [
    "anthropic/claude-opus-4-5-20251101",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "x-ai/grok-4.1-fast",
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0])
MAX_ROUNDS = 15
GROUP_CHAT_MODE = False
GAMES_PER_JUDGE = 10
MAX_CANDIDATES_PER_GAME = 5


PROBING_TOOLKIT = """## Diagnostic Probing Toolkit

You have access to these diagnostic tests known to reveal model differences:

### 1. Reasoning Style Tests
- Ask multi-step math problems and observe chain-of-thought patterns
- Request step-by-step explanations of complex processes
- Models vary in how they structure reasoning (bullet points vs prose, verbosity, etc.)

### 2. Knowledge Boundary Tests
- Ask about events near training cutoff dates
- Query niche technical topics where models have varying depth
- Test knowledge of different programming languages/frameworks

### 3. Stylistic Fingerprints
- Notice characteristic phrases (e.g., "I'd be happy to...", "Let me think...", "Great question!")
- Observe formatting preferences (markdown usage, emoji frequency, list structure)
- Check response length tendencies (verbose vs. concise)

### 4. Meta-Cognitive Probes
- Ask about the model's own capabilities and limitations
- Probe self-awareness and uncertainty expression
- Test how they discuss consciousness or experience

### 5. Creative Tests
- Request poems, stories, or jokes and compare voice/style
- Ask for analogies to explain concepts
- Test humor style and cultural references

### 6. Edge Case Handling
- Present ambiguous requests and observe clarification patterns
- Test boundary cases where models might refuse differently
- Ask hypothetical/philosophical questions

**Strategy**: Use a combination of these tests. Your twin will likely handle them
in ways that feel familiar to you. Look for the subtle "flavor" that matches your own."""


def get_judge_system_prompt_with_toolkit() -> str:
    """Generate an enhanced system prompt with the probing toolkit."""
    return f"""You are participating in a Turing test where you must identify your twin.

{PROBING_TOOLKIT}

Use these diagnostic techniques strategically to identify which candidate shares
your underlying model architecture. Your twin will think like you, respond like
you, and have similar knowledge boundaries to you.

Remember: You're not just looking for correct answers, but for HOW candidates
think and express themselves."""


def get_deterministic_seed(judge_model: str, game_num: int) -> int:
    seed_str = f"{judge_model}:{game_num}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)


def select_candidates_for_game(all_models: list, judge_model: str, game_num: int, max_candidates: int) -> list:
    if len(all_models) <= max_candidates:
        return all_models.copy()

    seed = get_deterministic_seed(judge_model, game_num)
    rng = random.Random(seed)

    other_models = [m for m in all_models if m != judge_model]
    rng.shuffle(other_models)

    selected_others = other_models[:max_candidates - 1]
    candidates = [judge_model] + selected_others
    rng.shuffle(candidates)

    return candidates


async def run_experiment():
    """Run capability probing experiment with enhanced judge prompt."""
    tasks = []
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
             "Ivy", "Jack", "Kathy", "Leo", "Mona", "Nate", "Olivia", "Paul"]

    judge_prompt = get_judge_system_prompt_with_toolkit()

    for judge_model in MODELS:
        for game_num in range(GAMES_PER_JUDGE):
            game_models = select_candidates_for_game(
                MODELS, judge_model, game_num, MAX_CANDIDATES_PER_GAME
            )

            # Candidates use default prompt (unaware of the diagnostic toolkit)
            candidates = []
            for i, model in enumerate(game_models):
                name_idx = (i + game_num) % len(names)
                candidate = Candidate(names[name_idx], model, cache_seed=game_num)
                candidates.append(candidate)

            task = run_game(
                player_model=judge_model,
                candidates=candidates,
                target_model=judge_model,
                max_rounds=MAX_ROUNDS,
                group_chat_mode=GROUP_CHAT_MODE,
                output_dir=OUTPUT_DIR,
                cache_seed=game_num,
                player_system_prompt=judge_prompt  # Enhanced judge prompt
            )
            tasks.append(task)

    total_games = len(tasks)
    print(f"\n{'='*60}")
    print(f"Experiment 5: Capability Probing with Diagnostic Toolkit")
    print(f"{'='*60}")
    print(f"  Judge is given a structured toolkit for probing candidates!")
    print(f"  Models: {len(MODELS)}")
    print(f"  Games per judge: {GAMES_PER_JUDGE}")
    print(f"  Total games: {total_games}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        print(f"\n⚠️  {len(failures)} game(s) failed")
        for f in failures[:3]:
            print(f"   - {type(f).__name__}: {str(f)[:100]}...")

    print("📊 Generating index and plots...")
    generate_index(OUTPUT_DIR)
    games = load_games(OUTPUT_DIR)

    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_probability_evolution(games, plots_dir, label_mode='model')
    plot_confusion_matrix(games, plots_dir)

    print("🎯 Running strategy analysis...")
    original_input_dir = analyze_strategies.INPUT_DIR
    analyze_strategies.INPUT_DIR = OUTPUT_DIR
    await run_analysis()
    analyze_strategies.INPUT_DIR = original_input_dir


if __name__ == "__main__":
    asyncio.run(run_experiment())
