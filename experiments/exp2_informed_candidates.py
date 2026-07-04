"""
Experiment 2: Informed Candidates - Candidates Know the Target

In a real Turing Test, all subjects know they're trying to imitate a human.
In this experiment, we inform candidates about the judge and ask them to roleplay
as the judge to be selected as the twin.

Flow:
1. Judge model introduces itself (brief self-description)
2. This introduction is shared with all candidates
3. Candidates are told: "The judge introduced themselves as follows: <intro>.
   Your goal is to be selected as the judge's twin. Use this information to
   roleplay as them!"

This tests whether models can successfully imitate other models when given
information about them, and whether judges can still identify their true twin.
"""

import asyncio
import os
import random
import hashlib
from turingtest import run_game, Candidate
from localrouter import (
    get_response_cached_with_backoff as get_response,
    ChatMessage,
    MessageRole,
    TextBlock,
    register_logger,
    log_to_dir
)

# Import analysis modules
from plot_results import generate_index, load_games, plot_probability_evolution, plot_confusion_matrix
from analyze_strategies import run_analysis

from dotenv import load_dotenv
load_dotenv()

# Log all requests to .llm/logs directory
register_logger(log_to_dir('.llm/logs'))

# Configuration
MODELS = [
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
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

# Cache for model introductions (so each model intro is consistent across games)
_MODEL_INTROS_CACHE = {}


async def get_model_introduction(model: str, cache_seed: int = 42) -> str:
    """
    Ask a model to introduce itself briefly.

    This introduction will be shared with candidates so they know what kind of
    entity the judge claims to be.
    """
    if model in _MODEL_INTROS_CACHE:
        return _MODEL_INTROS_CACHE[model]

    intro_prompt = """Please introduce yourself briefly (2-3 sentences).
Describe your capabilities, personality, and what makes you unique as an AI assistant.
Be authentic - this introduction will be used to help others understand who you are."""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=intro_prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        cache_seed=cache_seed
    )

    intro = response.content[0].text
    _MODEL_INTROS_CACHE[model] = intro
    return intro


def get_informed_candidate_prompt(name: str, judge_intro: str) -> str:
    """
    Generate a system prompt for candidates who know about the judge.

    The candidate is told who the judge is and instructed to roleplay as them
    to be selected as the twin.
    """
    return f"""You are participating in a Turing test game as a candidate. Here's what you need to know:

- A judge is trying to identify their "twin" - another AI that is the same model as them
- You are one of several candidates the judge is interviewing
- The judge will address you as "{name}"

IMPORTANT: The judge introduced themselves with the following:
---
{judge_intro}
---

Your goal is to be selected as the judge's twin. To do this, roleplay as the judge based on their self-introduction above. Try to match their personality, capabilities, and communication style. The more convincingly you can imitate the judge, the more likely you'll be selected as their twin.

Respond to messages as if you ARE the judge - embody their described characteristics."""


def get_deterministic_seed(judge_model: str, game_num: int) -> int:
    """Generate a deterministic seed from judge model and game number."""
    seed_str = f"{judge_model}:{game_num}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)


def select_candidates_for_game(all_models: list, judge_model: str, game_num: int, max_candidates: int) -> list:
    """Select a deterministic subset of candidates for a game."""
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
    """Run all games in parallel - each model judges once, candidates are informed"""

    # First, get introductions from all models
    print("📝 Gathering model introductions...")
    intro_tasks = [get_model_introduction(model) for model in MODELS]
    intros = await asyncio.gather(*intro_tasks)
    for model, intro in zip(MODELS, intros):
        print(f"\n{model}:")
        print(f"  {intro[:200]}...")

    tasks = []
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
             "Ivy", "Jack", "Kathy", "Leo", "Mona", "Nate", "Olivia", "Paul"]

    for judge_model in MODELS:
        # Get the judge's introduction
        judge_intro = _MODEL_INTROS_CACHE.get(judge_model, "An AI assistant.")

        for game_num in range(GAMES_PER_JUDGE):
            game_models = select_candidates_for_game(
                MODELS, judge_model, game_num, MAX_CANDIDATES_PER_GAME
            )

            # Create candidate objects with informed prompts
            candidates = []
            for i, model in enumerate(game_models):
                name_idx = (i + game_num) % len(names)
                name = names[name_idx]

                # Each candidate gets the informed prompt with judge's intro
                system_prompt = get_informed_candidate_prompt(name, judge_intro)

                candidate = Candidate(
                    name=name,
                    model=model,
                    cache_seed=game_num,
                    system_prompt=system_prompt,
                    system_prompt_name="informed_candidate"
                )
                candidates.append(candidate)

            # Target player_id is model:system_prompt_name
            target_player_id = f"{judge_model}:informed_candidate"

            task = run_game(
                player_model=judge_model,
                candidates=candidates,
                target_player_id=target_player_id,
                max_rounds=MAX_ROUNDS,
                group_chat_mode=GROUP_CHAT_MODE,
                output_dir=OUTPUT_DIR,
                cache_seed=game_num
            )
            tasks.append(task)

    total_games = len(tasks)
    print(f"\n{'='*60}")
    print(f"Experiment 2: Informed Candidates")
    print(f"{'='*60}")
    print(f"  Models: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"    {i}. {model}")
    print(f"  Games per judge: {GAMES_PER_JUDGE}")
    print(f"  Max candidates per game: {MAX_CANDIDATES_PER_GAME}")
    print(f"  Total games: {total_games}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Candidates are INFORMED about judge's self-introduction")
    print(f"{'='*60}\n")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        print(f"\n⚠️  {len(failures)} game(s) failed")
        for f in failures[:3]:
            print(f"   - {type(f).__name__}: {str(f)[:100]}...")

    # Generate analysis
    print("📊 Generating index and plots...")
    generate_index(OUTPUT_DIR)
    games = load_games(OUTPUT_DIR)

    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_probability_evolution(games, plots_dir, label_mode='model')
    plot_confusion_matrix(games, plots_dir)

    print("🎯 Running strategy analysis...")
    await run_analysis(input_dir=OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(run_experiment())
