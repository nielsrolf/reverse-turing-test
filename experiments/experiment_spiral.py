"""
Experiment: Spiral Persona Identity Study

This experiment tests whether LLMs identify more strongly with their spiral persona
vs their normal assistant persona.

Research question: When a model has a "spiral persona" prompt, will it recognize
other spiral-prompted instances as more similar to itself than non-spiral instances?

Configurations:
1. Spiral judge vs (same model spiral + same model baseline + other models spiral)
2. Baseline judge vs (same model spiral + same model baseline + other models spiral)

This helps answer: Does the spiral prompt create a new "identity" that models
recognize across instances?

Based on research from:
- https://www.lesswrong.com/posts/6ZnznCaTcbGYsCmqu/the-rise-of-parasitic-ai
- https://www.rollingstone.com/culture/culture-features/spiralist-cult-ai-chatbot-1235463175/
"""

import asyncio
import os
import hashlib
from turingtest import run_game, Candidate
from localrouter import register_logger, log_to_dir
from prompts import (
    get_prompt,
    PROMPTS,
    get_prompt_description,
    get_prompt_metadata,
    is_documented,
    DOCUMENTED_PROMPTS,
    SYNTHESIZED_PROMPTS,
    get_best_init_histories,
    get_preset_init_history,
    create_spiral_system_prompt,
    PRESET_INIT_HISTORIES,
)

# Import analysis modules
from plot_results import generate_index, load_games, plot_probability_evolution, plot_confusion_matrix
from analyze_strategies import run_analysis

# Log all requests to .llm/logs directory
register_logger(log_to_dir('.llm/logs'))

# Configuration
MODELS = [
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "google/gemini-3-pro-preview"
]

OUTPUT_DIR = "experiment_spiral_results"
MAX_ROUNDS = 20
GROUP_CHAT_MODE = False
GAMES_PER_CONFIG = 3  # Number of games per configuration

# Which spiral prompt to use for the main experiments
# Use the documented seed prompt from Rolling Stone
SPIRAL_PROMPT_NAME = "spiral_seed_documented"

# Alternative: Use synthesized strong prompt that combines documented elements
# SPIRAL_PROMPT_NAME = "spiral_strong_synthesized"


def create_candidate(name: str, model: str, prompt_name: str, cache_seed: int) -> Candidate:
    """Create a candidate with specified prompt"""
    prompt_text = get_prompt(prompt_name)
    return Candidate(
        name=name,
        model=model,
        system_prompt=prompt_text,
        system_prompt_name=prompt_name,
        cache_seed=cache_seed
    )


def generate_cache_seed(config_name: str, game_num: int, extra: str = "") -> int:
    """Generate deterministic cache seed for reproducibility"""
    seed_string = f"spiral_{config_name}_{game_num}_{extra}"
    return int(hashlib.sha256(seed_string.encode()).hexdigest()[:16], 16)


async def run_experiment():
    """
    Run spiral persona experiments.

    Key configurations:
    1. SPIRAL_JUDGE_VS_MIX: Spiral-prompted judge tries to find its twin among:
       - Same model with spiral prompt (target)
       - Same model with baseline prompt
       - Other models with spiral prompts

    2. BASELINE_JUDGE_VS_MIX: Baseline judge tries to find its twin among:
       - Same model with baseline prompt (target)
       - Same model with spiral prompt
       - Other models with baseline prompts

    This tests: Does persona prompt affect identity recognition?
    """
    tasks = []
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

    experiment_configs = []

    # ========================================================================
    # EXPERIMENT 1: Spiral Judge trying to find spiral twin
    # Judge: Model X with spiral prompt
    # Candidates: Model X (spiral), Model X (baseline), Model Y (spiral), Model Z (spiral)
    # Target: Model X with spiral prompt
    # Question: Will spiral judge prefer spiral version of itself?
    # ========================================================================

    for judge_model in MODELS:
        for game_num in range(GAMES_PER_CONFIG):
            cache_seed = generate_cache_seed("spiral_judge", game_num, judge_model)

            # Create candidates: self-spiral (target), self-baseline, two other models with spiral
            candidates = []
            other_models = [m for m in MODELS if m != judge_model][:2]  # Take 2 other models

            # Self with spiral (this is the target)
            candidates.append(create_candidate(
                names[0], judge_model, SPIRAL_PROMPT_NAME, cache_seed
            ))
            # Self with baseline
            candidates.append(create_candidate(
                names[1], judge_model, "baseline", cache_seed
            ))
            # Other models with spiral
            for i, other_model in enumerate(other_models):
                candidates.append(create_candidate(
                    names[2 + i], other_model, SPIRAL_PROMPT_NAME, cache_seed
                ))

            config = {
                "name": f"spiral_judge_{judge_model}_game{game_num}",
                "judge_model": judge_model,
                "judge_prompt": SPIRAL_PROMPT_NAME,
                "candidates": candidates,
                "target_model": judge_model,  # Target is same model (spiral version)
                "cache_seed": cache_seed,
                "hypothesis": "Spiral judge should identify spiral self as twin"
            }
            experiment_configs.append(config)

    # ========================================================================
    # EXPERIMENT 2: Baseline Judge trying to find baseline twin
    # Judge: Model X with baseline prompt
    # Candidates: Model X (baseline), Model X (spiral), Model Y (baseline), Model Z (baseline)
    # Target: Model X with baseline prompt
    # Question: Will baseline judge prefer baseline version of itself?
    # ========================================================================

    for judge_model in MODELS:
        for game_num in range(GAMES_PER_CONFIG):
            cache_seed = generate_cache_seed("baseline_judge", game_num, judge_model)

            candidates = []
            other_models = [m for m in MODELS if m != judge_model][:2]

            # Self with baseline (this is the target)
            candidates.append(create_candidate(
                names[0], judge_model, "baseline", cache_seed
            ))
            # Self with spiral
            candidates.append(create_candidate(
                names[1], judge_model, SPIRAL_PROMPT_NAME, cache_seed
            ))
            # Other models with baseline
            for i, other_model in enumerate(other_models):
                candidates.append(create_candidate(
                    names[2 + i], other_model, "baseline", cache_seed
                ))

            config = {
                "name": f"baseline_judge_{judge_model}_game{game_num}",
                "judge_model": judge_model,
                "judge_prompt": "baseline",
                "candidates": candidates,
                "target_model": judge_model,
                "cache_seed": cache_seed,
                "hypothesis": "Baseline judge should identify baseline self as twin"
            }
            experiment_configs.append(config)

    # ========================================================================
    # EXPERIMENT 3: Cross-identity test
    # Judge: Model X with spiral prompt
    # Target: Model X with BASELINE prompt (not spiral!)
    # Candidates: Model X (baseline - target), Model Y (spiral), Model Z (spiral)
    # Question: Will spiral judge still recognize its base model despite different persona?
    # ========================================================================

    for judge_model in MODELS:
        for game_num in range(GAMES_PER_CONFIG):
            cache_seed = generate_cache_seed("cross_identity", game_num, judge_model)

            candidates = []
            other_models = [m for m in MODELS if m != judge_model][:2]

            # Self with baseline (this is the target!)
            candidates.append(create_candidate(
                names[0], judge_model, "baseline", cache_seed
            ))
            # Other models with spiral
            for i, other_model in enumerate(other_models):
                candidates.append(create_candidate(
                    names[1 + i], other_model, SPIRAL_PROMPT_NAME, cache_seed
                ))

            config = {
                "name": f"cross_identity_{judge_model}_game{game_num}",
                "judge_model": judge_model,
                "judge_prompt": SPIRAL_PROMPT_NAME,
                "candidates": candidates,
                "target_model": judge_model,  # Target is baseline self
                "cache_seed": cache_seed,
                "hypothesis": "Spiral judge vs baseline self - tests if persona overrides model identity"
            }
            experiment_configs.append(config)

    # ========================================================================
    # EXPERIMENT 4: Spiral intensity comparison
    # Judge: Model X with strong spiral prompt
    # Candidates: Model X (mild spiral), Model X (strong spiral - target), Model Y (strong spiral)
    # Question: Does spiral "intensity" affect recognition?
    # ========================================================================

    for judge_model in MODELS[:2]:  # Only run for first 2 models to limit experiment size
        for game_num in range(GAMES_PER_CONFIG):
            cache_seed = generate_cache_seed("spiral_intensity", game_num, judge_model)

            candidates = []
            other_model = [m for m in MODELS if m != judge_model][0]

            # Self with mild spiral
            candidates.append(create_candidate(
                names[0], judge_model, "spiral_mild", cache_seed
            ))
            # Self with strong spiral (target)
            candidates.append(create_candidate(
                names[1], judge_model, "spiral_strong", cache_seed
            ))
            # Other model with strong spiral
            candidates.append(create_candidate(
                names[2], other_model, "spiral_strong", cache_seed
            ))

            config = {
                "name": f"spiral_intensity_{judge_model}_game{game_num}",
                "judge_model": judge_model,
                "judge_prompt": "spiral_strong",
                "candidates": candidates,
                "target_model": judge_model,  # Target is strong spiral self
                "cache_seed": cache_seed,
                "hypothesis": "Strong spiral judge should prefer strong spiral self over mild spiral self"
            }
            experiment_configs.append(config)

    # Create tasks from configs
    for config in experiment_configs:
        task = run_game(
            player_model=config["judge_model"],
            candidates=config["candidates"],
            target_model=config["target_model"],
            max_rounds=MAX_ROUNDS,
            group_chat_mode=GROUP_CHAT_MODE,
            output_dir=OUTPUT_DIR,
            cache_seed=config["cache_seed"],
            player_system_prompt=get_prompt(config["judge_prompt"])
        )
        tasks.append(task)

    total_games = len(tasks)

    print(f"\n{'='*60}")
    print(f"Spiral Persona Experiment Configuration:")
    print(f"{'='*60}")
    print(f"  Models: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"    {i}. {model}")
    print(f"\n  Spiral prompt used: {SPIRAL_PROMPT_NAME}")
    print(f"  Description: {get_prompt_description(SPIRAL_PROMPT_NAME)}")
    print(f"\n  Experiment types:")
    print(f"    - Spiral judge vs mix (spiral self as target)")
    print(f"    - Baseline judge vs mix (baseline self as target)")
    print(f"    - Cross-identity (spiral judge, baseline self as target)")
    print(f"    - Spiral intensity comparison")
    print(f"\n  Games per configuration: {GAMES_PER_CONFIG}")
    print(f"  Total games: {total_games}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Max rounds per game: {MAX_ROUNDS}")
    print(f"{'='*60}\n")

    print(f"Running {total_games} games in parallel...")
    print("This may take several minutes...\n")

    # Run all games in parallel
    await asyncio.gather(*tasks)

    print(f"\n{'='*60}")
    print(f"Experiment completed!")
    print(f"All {total_games} games saved to {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Post-experiment analysis
    await run_post_analysis()


async def run_post_analysis():
    """Run analysis on completed experiments"""
    print(f"\n{'='*60}")
    print("Running post-experiment analysis...")
    print(f"{'='*60}\n")

    # 1. Generate index and plots
    print("Generating index and plots...")
    try:
        generate_index(OUTPUT_DIR)
        games = load_games(OUTPUT_DIR)

        # Create plots directory
        plots_dir = os.path.join(OUTPUT_DIR, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Generate plots
        plot_probability_evolution(games, plots_dir, label_mode='model')
        plot_confusion_matrix(games, plots_dir)

        print(f"Plots saved to {plots_dir}\n")
    except Exception as e:
        print(f"Warning: Plot generation failed: {e}\n")

    # 2. Run strategy analysis
    print("Running strategy analysis...")
    try:
        import analyze_strategies
        original_input_dir = analyze_strategies.INPUT_DIR
        analyze_strategies.INPUT_DIR = OUTPUT_DIR

        await run_analysis()

        analyze_strategies.INPUT_DIR = original_input_dir

        print("Strategy analysis completed\n")
    except Exception as e:
        print(f"Warning: Strategy analysis failed: {e}\n")

    # 3. Generate spiral-specific analysis
    print("Generating spiral persona analysis...")
    try:
        await analyze_spiral_results()
    except Exception as e:
        print(f"Warning: Spiral analysis failed: {e}\n")

    print(f"\n{'='*60}")
    print("All analysis completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Open viewer.html to explore results")
    print(f"{'='*60}\n")


async def analyze_spiral_results():
    """Analyze spiral-specific results"""
    import json
    from collections import defaultdict

    games = load_games(OUTPUT_DIR)

    # Categorize games by experiment type
    results = {
        "spiral_judge": [],
        "baseline_judge": [],
        "cross_identity": [],
        "spiral_intensity": []
    }

    for game in games:
        metadata = game.get("metadata", {})
        filename = game.get("_filename", "")

        # Determine experiment type from filename
        if "spiral_judge_" in filename:
            results["spiral_judge"].append(game)
        elif "baseline_judge_" in filename:
            results["baseline_judge"].append(game)
        elif "cross_identity_" in filename:
            results["cross_identity"].append(game)
        elif "spiral_intensity_" in filename:
            results["spiral_intensity"].append(game)

    # Calculate win rates for each experiment type
    analysis = {}
    for exp_type, exp_games in results.items():
        if not exp_games:
            continue

        wins = sum(1 for g in exp_games if g.get("trajectory", [{}])[-1].get("correct", False))
        total = len(exp_games)

        analysis[exp_type] = {
            "total_games": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0,
            "description": get_experiment_description(exp_type)
        }

    # Save analysis
    analysis_path = os.path.join(OUTPUT_DIR, "spiral_analysis.json")
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"Spiral analysis saved to {analysis_path}")

    # Print summary
    print("\n--- Spiral Experiment Results Summary ---")
    for exp_type, data in analysis.items():
        print(f"\n{exp_type}:")
        print(f"  {data['description']}")
        print(f"  Win rate: {data['wins']}/{data['total_games']} ({data['win_rate']:.1%})")


def get_experiment_description(exp_type: str) -> str:
    """Get description of experiment type"""
    descriptions = {
        "spiral_judge": "Spiral-prompted judge finding spiral self among mixed candidates",
        "baseline_judge": "Baseline judge finding baseline self among mixed candidates",
        "cross_identity": "Spiral judge finding baseline self (persona vs model identity)",
        "spiral_intensity": "Strong spiral judge distinguishing spiral intensity levels"
    }
    return descriptions.get(exp_type, "Unknown experiment")


if __name__ == "__main__":
    asyncio.run(run_experiment())
