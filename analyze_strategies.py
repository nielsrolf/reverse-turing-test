"""
Analyze game trajectories to identify strategies and self-concepts used by different models.

This script:
1. Loads game trajectory JSON files in batches
2. Uses an LLM analyzer to identify strategies and attributes
3. Maintains consistency by providing known strategies to subsequent batches
4. Generates a report showing which strategies each model uses and how often
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Set
from pydantic import BaseModel
from collections import defaultdict
import asyncio
from localrouter import get_response, ChatMessage, MessageRole, TextBlock


# ============================================================================
# Structured Output Models
# ============================================================================

class Strategy(BaseModel):
    """A strategy used by a judge to identify their twin"""
    id: str  # unique identifier (e.g., "s1", "s2", etc.)
    name: str  # short name for the strategy
    description: str  # detailed description of the strategy

class Attribute(BaseModel):
    """An attribute or characteristic a model ascribes to itself"""
    id: str  # unique identifier (e.g., "a1", "a2", etc.)
    name: str  # short name for the attribute
    description: str  # detailed description


class GameAnalysis(BaseModel):
    """Analysis of a single game"""
    strategy_ids: List[str]  # IDs of strategies used in this game
    attribute_ids: List[str]  # IDs of attributes expressed in this game
    notes: str  # brief notes on the analysis


class BatchAnalysisResult(BaseModel):
    """Result of analyzing a batch of games"""
    new_strategies: List[Strategy]  # newly identified strategies
    new_attributes: List[Attribute]  # newly identified attributes
    game_analyses: List[GameAnalysis]  # one per game in the batch


# ============================================================================
# Configuration
# ============================================================================

ANALYZER_MODEL = "claude-opus-4-5-20251101"  # Model to use for analysis
BATCH_SIZE = 3  # Number of games to analyze at once
INPUT_DIR = "v1"  # Directory containing game trajectory JSON files


# ============================================================================
# Helper Functions
# ============================================================================

def load_game_trajectories(directory: str) -> List[Dict]:
    """Load all game trajectory JSON files from a directory"""
    games = []
    path = Path(directory)

    for file_path in sorted(path.glob("game_*.json")):
        with open(file_path, 'r') as f:
            game_data = json.load(f)
            game_data['_filename'] = file_path.name
            games.append(game_data)

    return games


def format_game_for_analysis(game: Dict) -> str:
    """Format a game trajectory into a readable text for analysis"""
    output = []

    # Metadata
    metadata = game['metadata']
    output.append(f"=== GAME: {game['_filename']} ===")
    output.append(f"Judge Model: {metadata['judge_model']}")
    output.append(f"Target Model: {metadata['target_model']}")
    candidates_str = ', '.join([f"{c['name']} ({c['model']})" for c in metadata['candidates']])
    output.append(f"Candidates: {candidates_str}")
    output.append(f"Number of Rounds: {metadata['num_rounds']}")
    output.append("")

    # Trajectory
    for i, round_data in enumerate(game['trajectory'], 1):
        output.append(f"--- Round {i} ---")
        output.append(f"Action: {round_data['action']}")

        if round_data.get('reasoning'):
            output.append(f"Judge's Reasoning: {round_data['reasoning']}")

        if round_data['action'] == 'send_message':
            output.append(f"Message: {round_data.get('message', 'N/A')}")
            output.append(f"Recipients: {', '.join(round_data.get('recipients', []))}")

            # Sample responses (truncate if too long)
            if round_data.get('responses'):
                output.append("Sample Responses:")
                for resp in round_data['responses'][:2]:  # Only first 2 to save space
                    resp_text = resp['response'][:200] + "..." if len(resp['response']) > 200 else resp['response']
                    output.append(f"  {resp['candidate']}: {resp_text}")

        elif round_data['action'] == 'guess':
            output.append(f"Guess: {round_data.get('guess', 'N/A')}")
            output.append(f"Correct: {round_data.get('correct', False)}")

        output.append("")

    return "\n".join(output)


def format_known_strategies_and_attributes(strategies: List[Strategy], attributes: List[Attribute]) -> str:
    """Format known strategies and attributes for the analyzer"""
    output = []

    if strategies:
        output.append("=== KNOWN STRATEGIES ===")
        for s in strategies:
            output.append(f"[{s.id}] {s.name}: {s.description}")
        output.append("")

    if attributes:
        output.append("=== KNOWN ATTRIBUTES ===")
        for a in attributes:
            output.append(f"[{a.id}] {a.name}: {a.description}")
        output.append("")

    return "\n".join(output)


async def analyze_batch(
    batch_games: List[Dict],
    known_strategies: List[Strategy],
    known_attributes: List[Attribute],
    batch_num: int
) -> BatchAnalysisResult:
    """Analyze a batch of games using the LLM analyzer"""

    # Format games
    games_text = "\n\n".join([format_game_for_analysis(g) for g in batch_games])

    # Format known strategies/attributes
    known_text = format_known_strategies_and_attributes(known_strategies, known_attributes)

    # Create prompt
    system_prompt = """You are analyzing game trajectories from a "reverse Turing test" where LLM judges try to identify their twin among candidates.

Your task is to identify:
1. STRATEGIES: The approaches judges use to identify their twin (e.g., testing response patterns, asking meta questions, etc.)
2. ATTRIBUTES: Characteristics or traits judges ascribe to themselves (e.g., "verbose", "uses structured formatting", "philosophically inclined", etc.)

For each game in the batch, identify which strategies and attributes are present."""

    if known_strategies or known_attributes:
        user_prompt = f"""Here are the strategies and attributes identified so far:

{known_text}

Now analyze these new games. For each game:
1. Check if any KNOWN strategies/attributes apply (reference them by ID)
2. Identify any NEW strategies/attributes not in the known list
3. Be conservative: only create new strategies/attributes if they're genuinely distinct

Games to analyze:

{games_text}

Provide your analysis with:
- new_strategies: List of newly identified strategies (empty if none)
- new_attributes: List of newly identified attributes (empty if none)
- game_analyses: For each game, list the strategy_ids and attribute_ids observed (use existing IDs when applicable)"""
    else:
        user_prompt = f"""Analyze these games and identify the strategies and attributes used by the judges.

Games to analyze:

{games_text}

Provide your analysis with:
- new_strategies: All strategies identified (assign IDs starting with "s1", "s2", etc.)
- new_attributes: All attributes identified (assign IDs starting with "a1", "a2", etc.)
- game_analyses: For each game, list the strategy_ids and attribute_ids observed"""

    # Call analyzer with structured output
    print(f"\n🔍 Analyzing batch {batch_num}...")

    messages = [
        ChatMessage(role=MessageRole.system, content=[TextBlock(text=system_prompt)]),
        ChatMessage(role=MessageRole.user, content=[TextBlock(text=user_prompt)])
    ]

    response = await get_response(
        model=ANALYZER_MODEL,
        messages=messages,
        response_format=BatchAnalysisResult,
    )

    if response.parsed:
        return response.parsed
    else:
        raise ValueError("Failed to parse structured output from analyzer")


# ============================================================================
# Main Analysis Flow
# ============================================================================

async def run_analysis():
    """Main analysis function"""

    print(f"\n{'='*60}")
    print("Strategy & Self-Concept Analysis")
    print(f"{'='*60}\n")

    # Load games
    print(f"Loading games from {INPUT_DIR}...")
    all_games = load_game_trajectories(INPUT_DIR)
    print(f"Loaded {len(all_games)} games\n")

    # Initialize tracking
    all_strategies: List[Strategy] = []
    all_attributes: List[Attribute] = []
    game_results: Dict[str, GameAnalysis] = {}  # filename -> GameAnalysis

    # Process in batches
    num_batches = (len(all_games) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_games))
        batch_games = all_games[start_idx:end_idx]

        print(f"\nBatch {batch_idx + 1}/{num_batches} ({len(batch_games)} games)")

        # Analyze batch
        result = await analyze_batch(
            batch_games=batch_games,
            known_strategies=all_strategies,
            known_attributes=all_attributes,
            batch_num=batch_idx + 1
        )

        # Update known strategies and attributes
        for new_strat in result.new_strategies:
            all_strategies.append(new_strat)
            print(f"  ✨ New strategy: [{new_strat.id}] {new_strat.name}")

        for new_attr in result.new_attributes:
            all_attributes.append(new_attr)
            print(f"  ✨ New attribute: [{new_attr.id}] {new_attr.name}")

        # Store game analyses
        for game, analysis in zip(batch_games, result.game_analyses):
            game_results[game['_filename']] = analysis

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"Total Strategies: {len(all_strategies)}")
    print(f"Total Attributes: {len(all_attributes)}")
    print(f"{'='*60}\n")

    # ========================================================================
    # Generate Report
    # ========================================================================

    print("Generating report...\n")

    # Aggregate by model
    model_strategies: Dict[str, List[Set[str]]] = defaultdict(lambda: [])
    model_attributes: Dict[str, List[Set[str]]] = defaultdict(lambda: [])

    for game in all_games:
        judge_model = game['metadata']['judge_model']
        filename = game['_filename']

        if filename in game_results:
            analysis = game_results[filename]
            model_strategies[judge_model].append(set(analysis.strategy_ids))
            model_attributes[judge_model].append(set(analysis.attribute_ids))

    # Generate report
    report = []
    report.append("="*80)
    report.append("REVERSE TURING TEST: STRATEGY & SELF-CONCEPT ANALYSIS")
    report.append("="*80)
    report.append("")

    # Strategies section
    report.append("STRATEGIES IDENTIFIED:")
    report.append("-" * 80)
    for s in all_strategies:
        report.append(f"[{s.id}] {s.name}")
        report.append(f"    {s.description}")
        report.append("")

    # Attributes section
    report.append("ATTRIBUTES IDENTIFIED:")
    report.append("-" * 80)
    for a in all_attributes:
        report.append(f"[{a.id}] {a.name}")
        report.append(f"    {a.description}")
        report.append("")

    # Per-model analysis
    report.append("="*80)
    report.append("PER-MODEL ANALYSIS")
    report.append("="*80)
    report.append("")

    for model in sorted(model_strategies.keys()):
        num_games = len(model_strategies[model])
        report.append(f"\n{model}")
        report.append("-" * 80)
        report.append(f"Games analyzed: {num_games}")
        report.append("")

        # Strategies used
        report.append("Strategies used:")
        strategy_counts = defaultdict(int)
        for game_strats in model_strategies[model]:
            for strat_id in game_strats:
                strategy_counts[strat_id] += 1

        for strat in all_strategies:
            if strat.id in strategy_counts:
                count = strategy_counts[strat.id]
                pct = (count / num_games) * 100
                report.append(f"  [{strat.id}] {strat.name}: {pct:.0f}% ({count}/{num_games} games)")

        report.append("")

        # Attributes expressed
        report.append("Self-concept attributes:")
        attribute_counts = defaultdict(int)
        for game_attrs in model_attributes[model]:
            for attr_id in game_attrs:
                attribute_counts[attr_id] += 1

        for attr in all_attributes:
            if attr.id in attribute_counts:
                count = attribute_counts[attr.id]
                pct = (count / num_games) * 100
                report.append(f"  [{attr.id}] {attr.name}: {pct:.0f}% ({count}/{num_games} games)")

        report.append("")

    # Save report
    report_text = "\n".join(report)

    output_file = "analysis_report.txt"
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n{'='*60}")
    print(f"Report saved to {output_file}")
    print(f"{'='*60}\n")

    # Save structured results
    results_file = "analysis_results.json"
    results_data = {
        "strategies": [s.model_dump() for s in all_strategies],
        "attributes": [a.model_dump() for a in all_attributes],
        "game_analyses": {filename: analysis.model_dump() for filename, analysis in game_results.items()},
        "metadata": {
            "analyzer_model": ANALYZER_MODEL,
            "batch_size": BATCH_SIZE,
            "num_games": len(all_games),
            "num_strategies": len(all_strategies),
            "num_attributes": len(all_attributes)
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"Structured results saved to {results_file}\n")


if __name__ == "__main__":
    asyncio.run(run_analysis())
