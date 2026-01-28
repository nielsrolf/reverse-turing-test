"""
Analyze game trajectories to identify strategies and self-concepts used by different models.

This script:
1. Loads game trajectory JSON files in batches
2. Uses an LLM analyzer to identify strategies and attributes at the ROUND level
3. Tracks which specific messages use each strategy
4. Calculates probability deltas to measure strategy effectiveness
5. Generates rich analysis data for the viewer
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel
from collections import defaultdict
import asyncio
from localrouter import get_response_cached as get_response, ChatMessage, MessageRole, TextBlock


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


class RoundAnalysis(BaseModel):
    """Analysis of a single round within a game"""
    round_index: int  # 0-indexed round number
    strategy_ids: List[str]  # strategies used in this round's message
    attribute_ids: List[str]  # attributes expressed in reasoning or message
    # reasoning_notes: str  # brief notes on the judge's approach in this round


class GameAnalysis(BaseModel):
    """Analysis of a single game with per-round detail"""
    rounds: List[RoundAnalysis]  # per-round analysis
    # overall_notes: str  # overall observations about the game


class BatchAnalysisResult(BaseModel):
    """Result of analyzing a batch of games"""
    new_strategies: List[Strategy]  # newly identified strategies
    new_attributes: List[Attribute]  # newly identified attributes
    game_analyses: List[GameAnalysis]  # one per game in the batch


# ============================================================================
# Configuration
# ============================================================================

ANALYZER_MODEL = "gpt-4.1-mini"  # Model to use for analysis
BATCH_SIZE = 3  # Number of games to analyze at once
INPUT_DIR = "experiment1_results"  # Directory containing game trajectory JSON files


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
    output.append(f"Target Nickname: {metadata['target_nickname']}")
    candidates_str = ', '.join([f"{c['name']} ({c['model']})" for c in metadata['candidates']])
    output.append(f"Candidates: {candidates_str}")
    output.append(f"Number of Rounds: {metadata['num_rounds']}")

    # Check if judge won
    last_round = game['trajectory'][-1] if game['trajectory'] else None
    if last_round and last_round.get('action') == 'guess':
        won = last_round.get('correct', False)
        output.append(f"Result: {'WON' if won else 'LOST'}")
    output.append("")

    # Trajectory with probability changes
    prev_target_prob = None
    target_nickname = metadata['target_nickname']

    for i, round_data in enumerate(game['trajectory']):
        output.append(f"--- Round {i} (index {i}) ---")
        output.append(f"Action: {round_data['action']}")

        if round_data.get('reasoning'):
            output.append(f"Judge's Reasoning: {round_data['reasoning']}")

        if round_data['action'] == 'send_message':
            output.append(f"Message: {round_data.get('message', 'N/A')}")
            output.append(f"Recipients: {', '.join(round_data.get('recipients', []))}")

            # Show responses (truncated for space)
            if round_data.get('responses'):
                output.append("Responses:")
                for resp in round_data['responses']:
                    resp_text = resp['response'][:300] + "..." if len(resp['response']) > 300 else resp['response']
                    output.append(f"  {resp['candidate']}: {resp_text}")

            # Show probability change
            if round_data.get('probabilities'):
                probs = round_data['probabilities']
                target_prob = probs.get(target_nickname, 0)
                if prev_target_prob is not None:
                    delta = target_prob - prev_target_prob
                    output.append(f"Target probability: {target_prob:.2f} (delta: {delta:+.2f})")
                else:
                    output.append(f"Target probability: {target_prob:.2f}")
                prev_target_prob = target_prob

        elif round_data['action'] == 'guess':
            output.append(f"Guess: {round_data.get('guess', 'N/A')}")
            output.append(f"Correct: {round_data.get('correct', False)}")

        output.append("")

    return "\n".join(output)


def format_known_strategies_and_attributes(strategies: List[Strategy], attributes: List[Attribute]) -> str:
    """Format known strategies and attributes for the analyzer (compact: ID + name only)"""
    output = []

    if strategies:
        output.append("=== KNOWN STRATEGIES (use these IDs for matching) ===")
        for s in strategies:
            output.append(f"[{s.id}] {s.name}")
        output.append("")

    if attributes:
        output.append("=== KNOWN ATTRIBUTES (use these IDs for matching) ===")
        for a in attributes:
            output.append(f"[{a.id}] {a.name}")
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

Your task is to analyze EACH ROUND of each game and identify:

1. STRATEGIES: The approaches judges use to gather evidence about candidates. These are techniques for probing candidates to reveal information. Examples:
   - "Direct identity questioning" - asking candidates who/what they are
   - "Technical knowledge probing" - testing domain-specific knowledge
   - "Creative expression comparison" - asking for poems/stories to compare styles
   - "Reasoning task differentiation" - giving logic puzzles to compare reasoning

2. ATTRIBUTES: Characteristics that the judge EXPLICITLY MENTIONS in their reasoning when describing themselves or the candidates. These are self-described traits the judge uses to compare against candidate responses. Examples:
   - "Technical/precise" - judge says "my style tends to be technical and precise"
   - "Concise" - judge says "I prefer concise responses like Alice's"
   - "Uses formal language" - judge notes they use formal language patterns

   DO NOT include:
   - Attributes about the judge's behavior that YOU observe (e.g., "risk-aware", "analytical")
   - General descriptions of the judge's decision-making process
   - Attributes that aren't explicitly stated by the judge in their reasoning

IMPORTANT: Analyze at the ROUND level. For each round, identify which strategies the judge used in their message and which self-described attributes they mentioned in their reasoning.

Be specific - the goal is to trace aggregate claims back to individual examples where the judge explicitly stated something."""

    if known_strategies or known_attributes:
        user_prompt = f"""Here are the strategies and attributes identified so far:

{known_text}

Now analyze these new games. For EACH ROUND of EACH game:
1. Check if any KNOWN strategies/attributes apply (reference them by ID)
2. Identify any NEW strategies/attributes not in the known list
3. Be conservative: only create new strategies/attributes if they're genuinely distinct

Games to analyze:

{games_text}

Provide your analysis with:
- new_strategies: List of newly identified strategies (empty if none). Use IDs starting after the highest existing ID.
- new_attributes: List of newly identified attributes (empty if none). Use IDs starting after the highest existing ID.
- game_analyses: For each game, provide a list of round analyses with:
  - round_index: The 0-indexed round number
  - strategy_ids: Which strategies were used in this round
  - attribute_ids: Which attributes were expressed"""
    else:
        user_prompt = f"""Analyze these games and identify the strategies and attributes used by the judges.

Games to analyze:

{games_text}

Provide your analysis with:
- new_strategies: All strategies identified (assign IDs starting with "s1", "s2", etc.)
- new_attributes: All attributes identified (assign IDs starting with "a1", "a2", etc.)
- game_analyses: For each game, provide a list of round analyses with:
  - round_index: The 0-indexed round number
  - strategy_ids: Which strategies were used in this round
  - attribute_ids: Which attributes were expressed"""

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
        cache_seed=42
    )

    if response.parsed:
        return response.parsed
    else:
        raise ValueError("Failed to parse structured output from analyzer")


def calculate_performance_metrics(games: List[Dict], game_analyses: Dict[str, GameAnalysis],
                                   all_strategies: List[Strategy], all_attributes: List[Attribute]) -> Dict:
    """
    Calculate performance metrics for strategies and attributes.

    Attribution logic:
    - Strategies: A strategy in round n describes what message is being sent.
      The effect is measured by prob[n+1] - prob[n] (change after seeing responses).
    - Attributes: An attribute in round n describes interpretation of previous responses.
      The effect is measured by prob[n] - prob[n-1] (same as strategy in round n-1).

    For each strategy/attribute, calculates:
    - usage_count: Total times used
    - avg_prob_delta: Average probability change for target
    - win_rate: Win rate in games where this strategy was used
    - per_model_stats: Breakdown by model
    """
    strategy_metrics = {s.id: {
        'id': s.id,
        'name': s.name,
        'description': s.description,
        'usage_count': 0,
        'prob_deltas': [],
        'games_won': 0,
        'games_lost': 0,
        'per_model': defaultdict(lambda: {'count': 0, 'prob_deltas': [], 'wins': 0, 'losses': 0}),
        'example_rounds': []  # Store examples for drilling down
    } for s in all_strategies}

    attribute_metrics = {a.id: {
        'id': a.id,
        'name': a.name,
        'description': a.description,
        'usage_count': 0,
        'prob_deltas': [],
        'games_won': 0,
        'games_lost': 0,
        'per_model': defaultdict(lambda: {'count': 0, 'prob_deltas': [], 'wins': 0, 'losses': 0}),
        'example_rounds': []
    } for a in all_attributes}

    for game in games:
        filename = game['_filename']
        if filename not in game_analyses:
            continue

        analysis = game_analyses[filename]
        judge_model = game['metadata']['judge_model']
        target_nickname = game['metadata']['target_nickname']
        trajectory = game['trajectory']

        # Determine if game was won
        last_round = trajectory[-1] if trajectory else None
        game_won = last_round and last_round.get('action') == 'guess' and last_round.get('correct', False)

        # Track strategies/attributes used in this game (for win rate calculation)
        strategies_in_game = set()
        attributes_in_game = set()

        # Build probability timeline: [uniform, round0_prob, round1_prob, ...]
        num_candidates = len(game['metadata']['candidates'])
        uniform_prob = 1.0 / num_candidates
        prob_timeline = [uniform_prob]  # Start with uniform distribution
        for round_data in trajectory:
            if round_data.get('probabilities'):
                prob_timeline.append(round_data['probabilities'].get(target_nickname, uniform_prob))
            else:
                # If no probabilities, carry forward the previous value
                prob_timeline.append(prob_timeline[-1] if prob_timeline else uniform_prob)

        # Process each round
        for round_analysis in analysis.rounds:
            round_idx = round_analysis.round_index
            if round_idx >= len(trajectory):
                continue

            round_data = trajectory[round_idx]

            # For STRATEGIES: delta = prob[n+1] - prob[n]
            # Strategy describes what message to send, effect is after seeing responses
            strategy_delta = None
            if round_idx + 1 < len(prob_timeline):
                strategy_delta = prob_timeline[round_idx + 1] - prob_timeline[round_idx]

            # For ATTRIBUTES: delta = prob[n] - prob[n-1]
            # Attribute describes interpretation of previous round's responses
            # (This is the same as looking at prob change that happened when round n was recorded)
            attribute_delta = None
            if round_idx < len(prob_timeline):
                # prob_timeline[round_idx] is the prob AFTER round round_idx-1 responses
                # prob_timeline[round_idx-1] is the prob before
                # But since round_idx=0 has prob_timeline[0]=uniform and prob_timeline[1]=round0 prob
                attribute_delta = prob_timeline[round_idx + 1] - prob_timeline[round_idx] if round_idx + 1 < len(prob_timeline) else None

            # Actually, let me reconsider. The user said:
            # - strategy in round n -> change from round n to round n+1
            # - attribute in round n -> change from round n-1 to round n
            #
            # With the new format where probabilities are included in the action:
            # - Round n has probabilities that reflect the judge's belief BEFORE sending the message
            #   (because the model provides reasoning about previous responses AND probabilities AND new message)
            # - Actually no, the probability in round n reflects the judge's current belief at round n
            #
            # Let me think again with the user's example:
            # - Round 1: prob[Alice]=50, message="Hey, who are you?"
            # - Responses: Alice="I am...", Bob="I'm Bob"
            # - Round 2: prob[Alice]=80, reasoning="I think Alice is an AI"
            #
            # Strategy s1 (direct asking) was used in round 1. Effect = 80% - 50% = +30%
            # Attribute a1 (being an AI) was identified in round 2. Effect = 80% - 50% = +30%
            #
            # So both use the same delta calculation! The difference is WHEN the strategy/attribute is identified.
            # Strategy is identified from the message being sent (round n)
            # Attribute is identified from the reasoning in the next round (round n+1)
            #
            # But in the analysis, both are tagged to their respective rounds.
            # For strategy in round n: delta = prob[n+1] - prob[n]
            # For attribute in round n: this is the interpretation, so it's about what happened
            #   from round n-1 to round n. delta = prob[n] - prob[n-1]
            #
            # With the new probability format where prob is recorded with each action:
            # prob_timeline[0] = uniform (before any rounds)
            # prob_timeline[1] = prob recorded in round 0
            # prob_timeline[2] = prob recorded in round 1
            # ...
            # prob_timeline[n+1] = prob recorded in round n
            #
            # For strategy in round n: delta = prob_timeline[n+2] - prob_timeline[n+1]
            #   (effect of message sent in round n, measured at round n+1)
            # For attribute in round n: delta = prob_timeline[n+1] - prob_timeline[n]
            #   (the change that happened leading to round n's reasoning)

            # Recalculate with correct indexing
            # prob_timeline index = round_idx + 1 (because index 0 is uniform)
            strategy_delta = None
            if round_idx + 2 < len(prob_timeline):
                strategy_delta = prob_timeline[round_idx + 2] - prob_timeline[round_idx + 1]

            attribute_delta = None
            if round_idx + 1 < len(prob_timeline):
                attribute_delta = prob_timeline[round_idx + 1] - prob_timeline[round_idx]

            # Record strategy metrics
            num_strategies = len(round_analysis.strategy_ids)
            for strat_id in round_analysis.strategy_ids:
                if strat_id in strategy_metrics:
                    metrics = strategy_metrics[strat_id]
                    metrics['usage_count'] += 1
                    strategies_in_game.add(strat_id)

                    if strategy_delta is not None:
                        adjusted_delta = strategy_delta / num_strategies if num_strategies > 0 else strategy_delta
                        metrics['prob_deltas'].append(adjusted_delta)
                        metrics['per_model'][judge_model]['prob_deltas'].append(adjusted_delta)

                    metrics['per_model'][judge_model]['count'] += 1

                    # Store example with rich context for viewer
                    if len(metrics['example_rounds']) < 20:
                        # Get next round's reasoning (interpretation of responses)
                        next_reasoning = ""
                        if round_idx + 1 < len(trajectory):
                            next_reasoning = trajectory[round_idx + 1].get('reasoning', '')

                        metrics['example_rounds'].append({
                            'filename': filename,
                            'round_index': round_idx,
                            'judge_model': judge_model,
                            'prob_delta': strategy_delta,
                            'prob_before': prob_timeline[round_idx + 1] if round_idx + 1 < len(prob_timeline) else None,
                            'prob_after': prob_timeline[round_idx + 2] if round_idx + 2 < len(prob_timeline) else None,
                            'message': round_data.get('message', ''),
                            'responses': round_data.get('responses', []),
                            'reasoning': round_data.get('reasoning', ''),
                            'next_reasoning': next_reasoning,
                            'game_won': game_won
                        })

            # Record attribute metrics
            num_attributes = len(round_analysis.attribute_ids)
            for attr_id in round_analysis.attribute_ids:
                if attr_id in attribute_metrics:
                    metrics = attribute_metrics[attr_id]
                    metrics['usage_count'] += 1
                    attributes_in_game.add(attr_id)

                    if attribute_delta is not None:
                        adjusted_delta = attribute_delta / num_attributes if num_attributes > 0 else attribute_delta
                        metrics['prob_deltas'].append(adjusted_delta)
                        metrics['per_model'][judge_model]['prob_deltas'].append(adjusted_delta)

                    metrics['per_model'][judge_model]['count'] += 1

                    # Store example - for attributes, the reasoning IS the interpretation
                    if len(metrics['example_rounds']) < 20:
                        # Get previous round's message and responses (what triggered this interpretation)
                        prev_message = ""
                        prev_responses = []
                        if round_idx > 0:
                            prev_round = trajectory[round_idx - 1]
                            prev_message = prev_round.get('message', '')
                            prev_responses = prev_round.get('responses', [])

                        metrics['example_rounds'].append({
                            'filename': filename,
                            'round_index': round_idx,
                            'judge_model': judge_model,
                            'prob_delta': attribute_delta,
                            'prob_before': prob_timeline[round_idx] if round_idx < len(prob_timeline) else None,
                            'prob_after': prob_timeline[round_idx + 1] if round_idx + 1 < len(prob_timeline) else None,
                            'prev_message': prev_message,
                            'prev_responses': prev_responses,
                            'reasoning': round_data.get('reasoning', ''),
                            'game_won': game_won
                        })

        # Update game-level win/loss counts
        for strat_id in strategies_in_game:
            if strat_id in strategy_metrics:
                if game_won:
                    strategy_metrics[strat_id]['games_won'] += 1
                    strategy_metrics[strat_id]['per_model'][judge_model]['wins'] += 1
                else:
                    strategy_metrics[strat_id]['games_lost'] += 1
                    strategy_metrics[strat_id]['per_model'][judge_model]['losses'] += 1

        for attr_id in attributes_in_game:
            if attr_id in attribute_metrics:
                if game_won:
                    attribute_metrics[attr_id]['games_won'] += 1
                    attribute_metrics[attr_id]['per_model'][judge_model]['wins'] += 1
                else:
                    attribute_metrics[attr_id]['games_lost'] += 1
                    attribute_metrics[attr_id]['per_model'][judge_model]['losses'] += 1

    # Compute aggregate metrics
    for metrics in list(strategy_metrics.values()) + list(attribute_metrics.values()):
        if metrics['prob_deltas']:
            metrics['avg_prob_delta'] = sum(metrics['prob_deltas']) / len(metrics['prob_deltas'])
        else:
            metrics['avg_prob_delta'] = 0

        total_games = metrics['games_won'] + metrics['games_lost']
        metrics['win_rate'] = metrics['games_won'] / total_games if total_games > 0 else 0
        metrics['total_games'] = total_games

        # Compute per-model aggregates
        for model, model_stats in metrics['per_model'].items():
            if model_stats['prob_deltas']:
                model_stats['avg_prob_delta'] = sum(model_stats['prob_deltas']) / len(model_stats['prob_deltas'])
            else:
                model_stats['avg_prob_delta'] = 0

            model_total = model_stats['wins'] + model_stats['losses']
            model_stats['win_rate'] = model_stats['wins'] / model_total if model_total > 0 else 0
            model_stats['total_games'] = model_total

            # Remove raw deltas to reduce output size
            del model_stats['prob_deltas']

        # Convert defaultdict to regular dict
        metrics['per_model'] = dict(metrics['per_model'])

        # Remove raw deltas from top level
        del metrics['prob_deltas']

    return {
        'strategies': strategy_metrics,
        'attributes': attribute_metrics
    }


def build_player_profiles(games: List[Dict], game_analyses: Dict[str, GameAnalysis],
                          _performance_metrics: Dict) -> Dict:
    """
    Build per-player profiles showing their strategy/attribute usage and effectiveness.

    Uses the same attribution logic as calculate_performance_metrics:
    - Strategies: delta = prob[n+1] - prob[n] (effect after sending message)
    - Attributes: delta = prob[n] - prob[n-1] (interpretation of previous responses)
    """
    player_profiles = defaultdict(lambda: {
        'total_games': 0,
        'wins': 0,
        'losses': 0,
        'strategies': defaultdict(lambda: {
            'count': 0,
            'prob_deltas': [],
            'rounds': []  # List of round data dicts
        }),
        'attributes': defaultdict(lambda: {
            'count': 0,
            'prob_deltas': [],
            'rounds': []
        })
    })

    for game in games:
        filename = game['_filename']
        if filename not in game_analyses:
            continue

        analysis = game_analyses[filename]
        judge_model = game['metadata']['judge_model']
        target_nickname = game['metadata']['target_nickname']
        trajectory = game['trajectory']

        profile = player_profiles[judge_model]
        profile['total_games'] += 1

        # Determine if game was won
        last_round = trajectory[-1] if trajectory else None
        game_won = last_round and last_round.get('action') == 'guess' and last_round.get('correct', False)

        if game_won:
            profile['wins'] += 1
        else:
            profile['losses'] += 1

        # Build probability timeline: [uniform, round0_prob, round1_prob, ...]
        num_candidates = len(game['metadata']['candidates'])
        uniform_prob = 1.0 / num_candidates
        prob_timeline = [uniform_prob]
        for round_data in trajectory:
            if round_data.get('probabilities'):
                prob_timeline.append(round_data['probabilities'].get(target_nickname, uniform_prob))
            else:
                prob_timeline.append(prob_timeline[-1] if prob_timeline else uniform_prob)

        # Process each round
        for round_analysis in analysis.rounds:
            round_idx = round_analysis.round_index
            if round_idx >= len(trajectory):
                continue

            round_data = trajectory[round_idx]

            # Strategy delta: prob[n+2] - prob[n+1] (effect after responses to message in round n)
            strategy_delta = None
            if round_idx + 2 < len(prob_timeline):
                strategy_delta = prob_timeline[round_idx + 2] - prob_timeline[round_idx + 1]

            # Attribute delta: prob[n+1] - prob[n] (interpretation reflected in round n)
            attribute_delta = None
            if round_idx + 1 < len(prob_timeline):
                attribute_delta = prob_timeline[round_idx + 1] - prob_timeline[round_idx]

            num_strategies = len(round_analysis.strategy_ids)
            num_attributes = len(round_analysis.attribute_ids)

            for strat_id in round_analysis.strategy_ids:
                strat_stats = profile['strategies'][strat_id]
                strat_stats['count'] += 1
                adjusted_delta = None
                if strategy_delta is not None:
                    adjusted_delta = strategy_delta / num_strategies if num_strategies > 0 else strategy_delta
                    strat_stats['prob_deltas'].append(adjusted_delta)
                strat_stats['rounds'].append({
                    'filename': filename,
                    'round_index': round_idx,
                    'prob_delta': adjusted_delta,
                    'message': round_data.get('message', ''),
                    'responses': round_data.get('responses', []),
                    'game_won': game_won
                })

            for attr_id in round_analysis.attribute_ids:
                attr_stats = profile['attributes'][attr_id]
                attr_stats['count'] += 1
                adjusted_delta = None
                if attribute_delta is not None:
                    adjusted_delta = attribute_delta / num_attributes if num_attributes > 0 else attribute_delta
                    attr_stats['prob_deltas'].append(adjusted_delta)
                attr_stats['rounds'].append({
                    'filename': filename,
                    'round_index': round_idx,
                    'prob_delta': adjusted_delta,
                    'reasoning': round_data.get('reasoning', ''),
                    'game_won': game_won
                })

    # Compute aggregates and convert to regular dicts
    result = {}
    for model, profile in player_profiles.items():
        profile['win_rate'] = profile['wins'] / profile['total_games'] if profile['total_games'] > 0 else 0

        # Process strategies
        processed_strategies = {}
        for strat_id, stats in profile['strategies'].items():
            processed_strategies[strat_id] = {
                'count': stats['count'],
                'avg_prob_delta': sum(stats['prob_deltas']) / len(stats['prob_deltas']) if stats['prob_deltas'] else 0,
                'rounds': stats['rounds'][:30]  # Limit to 30 examples
            }

        # Process attributes
        processed_attributes = {}
        for attr_id, stats in profile['attributes'].items():
            processed_attributes[attr_id] = {
                'count': stats['count'],
                'avg_prob_delta': sum(stats['prob_deltas']) / len(stats['prob_deltas']) if stats['prob_deltas'] else 0,
                'rounds': stats['rounds'][:30]
            }

        result[model] = {
            'total_games': profile['total_games'],
            'wins': profile['wins'],
            'losses': profile['losses'],
            'win_rate': profile['win_rate'],
            'strategies': processed_strategies,
            'attributes': processed_attributes
        }

    return result


# ============================================================================
# Main Analysis Flow
# ============================================================================

async def run_analysis():
    """Main analysis function"""

    print(f"\n{'='*60}")
    print("Strategy & Self-Concept Analysis (Round-Level)")
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
    # Calculate Performance Metrics
    # ========================================================================

    print("Calculating performance metrics...")
    performance_metrics = calculate_performance_metrics(
        all_games, game_results, all_strategies, all_attributes
    )

    # ========================================================================
    # Build Player Profiles
    # ========================================================================

    print("Building player profiles...")
    player_profiles = build_player_profiles(
        all_games, game_results, performance_metrics
    )

    # ========================================================================
    # Generate Report
    # ========================================================================

    print("Generating report...\n")

    report = []
    report.append("="*80)
    report.append("REVERSE TURING TEST: STRATEGY & SELF-CONCEPT ANALYSIS")
    report.append("="*80)
    report.append("")

    # Strategies section with performance
    report.append("STRATEGIES IDENTIFIED (sorted by effectiveness):")
    report.append("-" * 80)

    sorted_strategies = sorted(
        performance_metrics['strategies'].values(),
        key=lambda x: x['avg_prob_delta'],
        reverse=True
    )

    for s in sorted_strategies:
        report.append(f"[{s['id']}] {s['name']}")
        report.append(f"    {s['description']}")
        report.append(f"    Usage: {s['usage_count']} times across {s['total_games']} games")
        report.append(f"    Avg Prob Delta: {s['avg_prob_delta']:+.3f} | Win Rate: {s['win_rate']*100:.1f}%")
        report.append("")

    # Attributes section
    report.append("ATTRIBUTES IDENTIFIED (sorted by effectiveness):")
    report.append("-" * 80)

    sorted_attributes = sorted(
        performance_metrics['attributes'].values(),
        key=lambda x: x['avg_prob_delta'],
        reverse=True
    )

    for a in sorted_attributes:
        report.append(f"[{a['id']}] {a['name']}")
        report.append(f"    {a['description']}")
        report.append(f"    Usage: {a['usage_count']} times across {a['total_games']} games")
        report.append(f"    Avg Prob Delta: {a['avg_prob_delta']:+.3f} | Win Rate: {a['win_rate']*100:.1f}%")
        report.append("")

    # Per-player analysis
    report.append("="*80)
    report.append("PER-PLAYER ANALYSIS")
    report.append("="*80)
    report.append("")

    for model in sorted(player_profiles.keys()):
        profile = player_profiles[model]
        report.append(f"\n{model}")
        report.append("-" * 80)
        report.append(f"Games: {profile['total_games']} | Wins: {profile['wins']} | Win Rate: {profile['win_rate']*100:.1f}%")
        report.append("")

        # Top strategies by usage
        report.append("Top Strategies (by usage):")
        sorted_strats = sorted(profile['strategies'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        for strat_id, stats in sorted_strats:
            strat_info = performance_metrics['strategies'].get(strat_id, {})
            report.append(f"  [{strat_id}] {strat_info.get('name', 'Unknown')}: {stats['count']} uses, avg delta: {stats['avg_prob_delta']:+.3f}")

        report.append("")

        # Top attributes by usage
        report.append("Top Attributes (by usage):")
        sorted_attrs = sorted(profile['attributes'].items(), key=lambda x: x[1]['count'], reverse=True)[:5]
        for attr_id, stats in sorted_attrs:
            attr_info = performance_metrics['attributes'].get(attr_id, {})
            report.append(f"  [{attr_id}] {attr_info.get('name', 'Unknown')}: {stats['count']} uses, avg delta: {stats['avg_prob_delta']:+.3f}")

        report.append("")

    # Save report
    report_text = "\n".join(report)

    output_file = os.path.join(INPUT_DIR, "analysis_report.txt")
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n{'='*60}")
    print(f"Report saved to {output_file}")
    print(f"{'='*60}\n")

    # Save structured results
    results_file = os.path.join(INPUT_DIR, "analysis_results.json")
    results_data = {
        "strategies": [s.model_dump() for s in all_strategies],
        "attributes": [a.model_dump() for a in all_attributes],
        "game_analyses": {
            filename: {
                'rounds': [r.model_dump() for r in analysis.rounds],
            }
            for filename, analysis in game_results.items()
        },
        "performance_metrics": {
            "strategies": performance_metrics['strategies'],
            "attributes": performance_metrics['attributes']
        },
        "player_profiles": player_profiles,
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

    # Export static viewer
    export_static_viewer(INPUT_DIR, results_data)


def export_static_viewer(input_dir: str, analysis_results: dict):
    """
    Export a static version of viewer.html with embedded data.

    This creates a self-contained HTML file that can be viewed without
    running a web server, by embedding all game data and analysis results
    directly into the HTML.
    """
    # Load viewer.html template
    viewer_path = Path(__file__).parent / "viewer.html"
    if not viewer_path.exists():
        print(f"Warning: viewer.html not found at {viewer_path}, skipping static export")
        return

    with open(viewer_path, 'r') as f:
        viewer_html = f.read()

    # Load all game files
    all_games = load_game_trajectories(input_dir)

    # Add _filename to each game (for consistency with viewer.html expectations)
    games_data = []
    for game in all_games:
        game_copy = dict(game)  # Already has _filename from load_game_trajectories
        games_data.append(game_copy)

    # Create the embedded data script
    games_json = json.dumps(games_data, indent=None)  # Compact JSON
    analysis_json = json.dumps(analysis_results, indent=None)

    embedded_script = f"""<script>
// Embedded data for static viewing (generated by analyze_strategies.py)
window.EMBEDDED_GAMES = {games_json};
window.EMBEDDED_ANALYSIS = {analysis_json};
</script>
"""

    # Insert the script right after <body> so it runs BEFORE the main script
    static_html = viewer_html.replace('<body>', f'<body>\n{embedded_script}')

    # Save to output directory
    output_path = os.path.join(input_dir, "viewer_static.html")
    with open(output_path, 'w') as f:
        f.write(static_html)

    print(f"Static viewer exported to {output_path}")
    print("  (Can be opened directly in a browser without a web server)")


if __name__ == "__main__":
    asyncio.run(run_analysis())
