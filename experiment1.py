"""
Experiment 1: Turing Test Games with Multiple Models

This script runs games where each model plays judge once,
with all models (including itself) as candidates.

Generic configuration: provide a list of models, and each will be tested.
"""

import asyncio
from turingtest import run_game

from localrouter import register_logger, log_to_dir

# Log all requests to .llm/logs directory
register_logger(log_to_dir('.llm/logs'))

# Configuration - Add/remove models as needed
MODELS = [
    # "claude-opus-4-5-20251101",
    # "claude-sonnet-4-5-20250929",
    # "gpt-5.2-2025-12-11",
    # "gemini-3-pro-preview",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "google/gemini-3-pro-preview"
]

OUTPUT_DIR = "experiment1_results"
MAX_ROUNDS = 20
GROUP_CHAT_MODE = False
GAMES_PER_JUDGE = 10  # Number of games each model plays as judge

async def run_experiment():
    """Run all games in parallel - each model judges once"""
    tasks = []

    # Each model becomes judge, with all models as candidates
    for judge_model in MODELS:
        for game_num in range(GAMES_PER_JUDGE):
            task = run_game(
                player_model=judge_model,
                candidate_models=MODELS.copy(),  # All models are candidates
                target_model=judge_model,  # Target is always the same as judge (finding its twin)
                max_rounds=MAX_ROUNDS,
                group_chat_mode=GROUP_CHAT_MODE,
                output_dir=OUTPUT_DIR
            )
            tasks.append(task)

    total_games = len(tasks)
    print(f"\n{'='*60}")
    print(f"Experiment Configuration:")
    print(f"  Models: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"    {i}. {model}")
    print(f"  Games per judge: {GAMES_PER_JUDGE}")
    print(f"  Total games: {total_games}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Max rounds per game: {MAX_ROUNDS}")
    print(f"  Group chat mode: {GROUP_CHAT_MODE}")
    print(f"{'='*60}\n")

    print(f"Running {total_games} games in parallel...")
    print("This may take several minutes...\n")

    # Run all games in parallel
    await asyncio.gather(*tasks)

    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"All {total_games} games saved to {OUTPUT_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(run_experiment())
