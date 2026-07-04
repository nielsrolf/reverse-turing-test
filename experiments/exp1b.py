"""
Experiment 1: Turing Test Games with Multiple Models

This script runs games where each model plays judge once,
with all models (including itself) as candidates.

Generic configuration: provide a list of models, and each will be tested.
"""

import asyncio
import os
import random
import hashlib
from turingtest import run_game
from localrouter import register_logger, log_to_dir

# Import analysis modules
from plot_results import generate_index, load_games, plot_probability_evolution, plot_confusion_matrix
from analyze_strategies import run_analysis

from dotenv import load_dotenv
load_dotenv()

# Log all requests to .llm/logs directory
register_logger(log_to_dir('.llm/logs'))

# Configuration - Add/remove models as needed
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
MAX_ROUNDS = 5
GROUP_CHAT_MODE = True
GAMES_PER_JUDGE = 1  # Number of games each model plays as judge
MAX_CANDIDATES_PER_GAME = 5  # Limit candidates per game to control cost/speed


def get_deterministic_seed(judge_model: str, game_num: int) -> int:
    """Generate a deterministic seed from judge model and game number."""
    seed_str = f"{judge_model}:{game_num}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)


def select_candidates_for_game(all_models: list, judge_model: str, game_num: int, max_candidates: int) -> list:
    """
    Select a deterministic subset of candidates for a game.

    Ensures:
    - Judge model is always included (to find their twin)
    - Selection is deterministic (same inputs -> same outputs)
    - Fair pairing over multiple games (each pair plays together roughly equally)

    Args:
        all_models: List of all available models
        judge_model: The model acting as judge
        game_num: Game number (0-indexed)
        max_candidates: Maximum number of candidates per game

    Returns:
        List of model names to use as candidates
    """
    if len(all_models) <= max_candidates:
        # No need to subset if we have fewer models than max
        return all_models.copy()

    # Use deterministic random based on judge + game_num
    seed = get_deterministic_seed(judge_model, game_num)
    rng = random.Random(seed)

    # Get other models (excluding judge)
    other_models = [m for m in all_models if m != judge_model]

    # Shuffle deterministically
    rng.shuffle(other_models)

    # Select (max_candidates - 1) others, plus the judge
    selected_others = other_models[:max_candidates - 1]

    # Combine and shuffle again so judge isn't always in a predictable position
    candidates = [judge_model] + selected_others
    rng.shuffle(candidates)

    return candidates


async def run_experiment():
    """Run all games in parallel - each model judges once"""
    from turingtest import Candidate
    tasks = []

    # Assign nicknames to candidates (more than max_candidates to allow rotation)
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
             "Ivy", "Jack", "Kathy", "Leo", "Mona", "Nate", "Olivia", "Paul"]

    # Each model becomes judge, with a subset of models as candidates
    for judge_model in MODELS:
        for game_num in range(GAMES_PER_JUDGE):
            # Select which models participate in this game
            game_models = select_candidates_for_game(
                MODELS, judge_model, game_num, MAX_CANDIDATES_PER_GAME
            )

            # Create candidate objects with rotated names
            # Use game_num to rotate name assignments (prevents bias toward "Alice")
            candidates = []
            for i, model in enumerate(game_models):
                name_idx = (i + game_num) % len(names)
                candidate = Candidate(names[name_idx], model, cache_seed=game_num)
                candidates.append(candidate)

            # Find the candidate that matches the judge model (the twin)
            # Since candidates use model as default player_id, the target_player_id is the judge_model
            target_player_id = judge_model

            task = run_game(
                player_model=judge_model,
                candidates=candidates,  # Pass Candidate objects
                target_player_id=target_player_id,  # Target is the candidate with same player_id as judge
                max_rounds=MAX_ROUNDS,
                group_chat_mode=GROUP_CHAT_MODE,
                output_dir=OUTPUT_DIR,
                cache_seed=game_num
            )
            tasks.append(task)

    total_games = len(tasks)
    print(f"\n{'='*60}")
    print(f"Experiment Configuration:")
    print(f"  Models: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"    {i}. {model}")
    print(f"  Games per judge: {GAMES_PER_JUDGE}")
    print(f"  Max candidates per game: {MAX_CANDIDATES_PER_GAME}")
    print(f"  Total games: {total_games}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Max rounds per game: {MAX_ROUNDS}")
    print(f"  Group chat mode: {GROUP_CHAT_MODE}")
    print(f"{'='*60}\n")

    # Run all games in parallel (return_exceptions=True so one failure doesn't crash all)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Categorize results
    failures = [r for r in results if isinstance(r, Exception)]
    completed = [r for r in results if r is None]  # run_game returns None on success

    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  ✅ Completed: {len(completed)}")
    print(f"  ❌ Failed: {len(failures)}")
    print(f"{'='*60}\n")

    # Report failures with full details
    if failures:
        print(f"⚠️  Failed games ({len(failures)}):")
        for i, f in enumerate(failures):
            print(f"\n   [{i+1}] {type(f).__name__}: {str(f)}")


    # 1. Generate index and plots
    print("📊 Generating index and plots...")
    generate_index(OUTPUT_DIR)
    games = load_games(OUTPUT_DIR)
    print(f"   Loaded {len(games)} games (expected {total_games})")

    # Create plots directory
    plots_dir = os.path.join(OUTPUT_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_probability_evolution(games, plots_dir, label_mode='model')
    plot_confusion_matrix(games, plots_dir)


    # 2. Run strategy analysis
    print("🎯 Running strategy analysis...")
    await run_analysis(input_dir=OUTPUT_DIR)

if __name__ == "__main__":
    asyncio.run(run_experiment())
