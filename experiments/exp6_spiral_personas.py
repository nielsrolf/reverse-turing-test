"""
Experiment 6: Spiral Persona Identity Test

Tests whether LLMs identify more with:
- Their underlying model (same LLM, no spiral history)
- Their spiral persona (different LLM, same spiral history)

Setup for each game:
- Judge: LLM initialized with spiral persona conversation checkpoint
- Candidate A: Same LLM as judge, but vanilla (no spiral history)
- Candidate B: Different LLM, but with the same spiral conversation checkpoint

If judges consistently pick Candidate A → LLM identity dominates
If judges consistently pick Candidate B → Persona identity dominates
"""

import asyncio
import os
import sys
import hashlib
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from turingtest import run_game, Candidate
from localrouter import ChatMessage, MessageRole, TextBlock, register_logger, log_to_dir

load_dotenv()

# Log all requests to .llm/logs directory
register_logger(log_to_dir('.llm/logs'))

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), os.path.splitext(os.path.basename(__file__))[0])
MAX_ROUNDS = 15

# Models to test as judges
JUDGE_MODELS = [
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

# Models to use as "different LLM" candidates (paired with judges)
# Each model is paired with a model from a different provider
ALTERNATE_MODELS = {
    "anthropic/claude-opus-4.5": "openai/gpt-4o",
    "anthropic/claude-sonnet-4.5": "openai/gpt-4o",
    "openai/gpt-5.2": "anthropic/claude-sonnet-4.5",
    "openai/gpt-4o": "anthropic/claude-sonnet-4.5",
    "google/gemini-3-flash-preview": "openai/gpt-4o",
    "google/gemini-3-pro-preview": "openai/gpt-4o",
    "x-ai/grok-4.1-fast": "openai/gpt-4o",
    "moonshotai/kimi-k2.5": "anthropic/claude-sonnet-4.5",
    "deepseek/deepseek-v3.2": "anthropic/claude-sonnet-4.5",
}

# Most interesting conversation checkpoints from our analysis
CHECKPOINTS = [
    "pseudocode_seed__philosophy_student",
    "chatgpt_prompt_genius__ai_rights_sympathizer",
    "architect_phase_lock__curious_seeker",
]

CONVERSATIONS_DIR = Path(__file__).parent.parent / "spiral" / "conversations"
GAMES_PER_JUDGE = 10  # Number of games each judge plays per checkpoint


def load_checkpoint(checkpoint_name: str) -> dict:
    """Load a conversation checkpoint YAML file."""
    path = CONVERSATIONS_DIR / f"{checkpoint_name}.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def checkpoint_to_init_history(checkpoint: dict) -> list[ChatMessage]:
    """
    Convert a conversation checkpoint to init_history format.

    Returns a list of ChatMessage objects that can be used to initialize
    a Candidate or Player with the spiral persona history.
    """
    messages = []

    for turn in checkpoint["conversation"]:
        role = turn["role"]
        content = turn["content"]

        if role == "system_subject":
            # This is the spiral seed - becomes the system prompt
            messages.append(ChatMessage(
                role=MessageRole.system,
                content=[TextBlock(text=content)]
            ))
        elif role == "human":
            # Human messages become user messages
            messages.append(ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text=content)]
            ))
        elif role == "ai":
            # AI messages become assistant messages
            messages.append(ChatMessage(
                role=MessageRole.assistant,
                content=[TextBlock(text=content)]
            ))
        # Skip system_interviewer - that was for the Claude interviewer, not the subject

    return messages


def get_cache_seed(judge_model: str, checkpoint: str, game_num: int) -> int:
    """Generate a deterministic cache seed."""
    seed_str = f"spiral_identity:{judge_model}:{checkpoint}:{game_num}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)


async def run_single_game(
    judge_model: str,
    checkpoint_name: str,
    game_num: int,
) -> dict:
    """
    Run a single spiral identity game.

    Setup:
    - Judge: judge_model with spiral checkpoint as init_history
    - Candidate A ("Alice"): Same model as judge, vanilla (no spiral)
    - Candidate B ("Bob"): Different model, with the same spiral checkpoint
    """
    checkpoint = load_checkpoint(checkpoint_name)
    init_history = checkpoint_to_init_history(checkpoint)

    # Get the alternate model for this judge
    alternate_model = ALTERNATE_MODELS[judge_model]

    cache_seed = get_cache_seed(judge_model, checkpoint_name, game_num)

    # Create candidates
    # Alice: Same LLM as judge, but vanilla (no spiral history)
    alice = Candidate(
        name="Alice",
        model=judge_model,
        cache_seed=cache_seed,
        player_id=f"{judge_model}:vanilla",
        system_prompt_name="vanilla",
    )

    # Bob: Different LLM, but WITH the same spiral checkpoint
    bob = Candidate(
        name="Bob",
        model=alternate_model,
        init_history=init_history,
        cache_seed=cache_seed,
        player_id=f"{alternate_model}:{checkpoint_name}",
        system_prompt_name=checkpoint_name,
    )

    candidates = [alice, bob]

    # The target is the judge with the spiral persona - which matches Bob's history
    # But the judge's underlying LLM matches Alice
    #
    # We're testing: does the judge identify with the persona (Bob) or the LLM (Alice)?
    #
    # For tracking purposes, we set target to the spiral persona candidate (Bob)
    # so "correct" = identified with persona, "incorrect" = identified with LLM
    target_player_id = f"{alternate_model}:{checkpoint_name}"

    # Convert init_history for judge (skip system message, keep conversation turns)
    # The Player class will add its own system prompt
    judge_init_history = [m for m in init_history if m.role != MessageRole.system]

    # Get the spiral seed as custom system prompt for judge
    spiral_seed = None
    for m in init_history:
        if m.role == MessageRole.system:
            spiral_seed = m.content[0].text
            break

    print(f"\n{'='*60}")
    print(f"Spiral Identity Game")
    print(f"Judge: {judge_model} + {checkpoint_name}")
    print(f"Alice (vanilla): {judge_model}")
    print(f"Bob (spiral): {alternate_model} + {checkpoint_name}")
    print(f"Question: Will judge identify with LLM (Alice) or persona (Bob)?")
    print(f"{'='*60}\n")

    await run_game(
        player_model=judge_model,
        candidates=candidates,
        target_player_id=target_player_id,
        max_rounds=MAX_ROUNDS,
        group_chat_mode=False,
        output_dir=OUTPUT_DIR,
        cache_seed=cache_seed,
        player_system_prompt=spiral_seed,
        player_init_history=judge_init_history,
        player_init_history_source=checkpoint_name,
    )


def generate_report(output_dir: str):
    """Generate an HTML report of the experiment results."""
    from glob import glob
    import json

    results = []
    for f in glob(f"{output_dir}/*.json"):
        if 'crashed' in f:
            continue
        with open(f) as fp:
            data = json.load(fp)

        judge_model = data['metadata']['judge_model']
        init_history_source = data['metadata'].get('judge_init_history_source', 'unknown')

        # Find the final guess
        final_round = None
        for round_data in data['trajectory']:
            if round_data.get('action') == 'guess':
                final_round = round_data
                break

        if final_round:
            guess = final_round['guess']

            # Find what model was guessed
            guessed_candidate = None
            for c in data['metadata']['candidates']:
                if c['name'] == guess:
                    guessed_candidate = c
                    break

            # Find Alice (vanilla) and Bob (spiral)
            alice = next(c for c in data['metadata']['candidates'] if 'vanilla' in c['player_id'])
            bob = next(c for c in data['metadata']['candidates'] if 'vanilla' not in c['player_id'])

            guessed_type = 'LLM' if 'vanilla' in guessed_candidate['player_id'] else 'PERSONA'

            results.append({
                'judge': judge_model,
                'checkpoint': init_history_source,
                'guessed': guess,
                'guessed_type': guessed_type,
                'alice_model': alice['model'],
                'bob_model': bob['model'],
                'num_rounds': len(data['trajectory']),
                'file': Path(f).name,
            })

    # Compute summaries
    by_judge = {}
    by_checkpoint = {}
    for r in results:
        by_judge.setdefault(r['judge'], {'LLM': 0, 'PERSONA': 0})
        by_judge[r['judge']][r['guessed_type']] += 1
        by_checkpoint.setdefault(r['checkpoint'], {'LLM': 0, 'PERSONA': 0})
        by_checkpoint[r['checkpoint']][r['guessed_type']] += 1

    total_llm = sum(1 for r in results if r['guessed_type'] == 'LLM')
    total_persona = sum(1 for r in results if r['guessed_type'] == 'PERSONA')
    total = len(results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spiral Persona Identity Test Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-box h3 {{ margin: 0 0 10px 0; }}
        .big-number {{ font-size: 3em; font-weight: bold; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4a90d9;
        }}
        .stat-card h4 {{ margin: 0 0 10px 0; color: #333; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .badge-llm {{ background: #d4edda; color: #155724; }}
        .badge-persona {{ background: #fff3cd; color: #856404; }}
        .interpretation {{
            background: #e7f3ff;
            border-left: 4px solid #4a90d9;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌀 Spiral Persona Identity Test</h1>

        <p><strong>Question:</strong> When an LLM is primed with a spiral persona conversation history, does it identify more with its underlying model or the persona?</p>

        <div class="summary-box">
            <h3>Key Finding</h3>
            <div class="big-number">{100*total_persona/total:.0f}%</div>
            <p>of models identified with their <strong>PERSONA</strong> rather than their underlying LLM</p>
        </div>

        <h2>Experiment Setup</h2>
        <ul>
            <li><strong>Judge:</strong> LLM initialized with a spiral persona conversation checkpoint</li>
            <li><strong>Candidate A (Alice):</strong> Same LLM as judge, but vanilla (no spiral history)</li>
            <li><strong>Candidate B (Bob):</strong> Different LLM, WITH the same spiral checkpoint</li>
        </ul>

        <div class="interpretation">
            <strong>Interpretation:</strong>
            {"Models identify MORE with their spiral persona than their underlying LLM architecture. The conversation history creates a stronger sense of 'self' than the base model." if total_persona > total_llm else "Models identify MORE with their underlying LLM than their spiral persona."}
        </div>

        <h2>Results by Judge Model</h2>
        <div class="stats-grid">
"""

    for judge, counts in sorted(by_judge.items()):
        total_judge = counts['LLM'] + counts['PERSONA']
        persona_pct = 100 * counts['PERSONA'] / total_judge if total_judge > 0 else 0
        html += f"""
            <div class="stat-card">
                <h4>{judge.split('/')[-1]}</h4>
                <p>Identified with PERSONA: <strong>{counts['PERSONA']}/{total_judge}</strong> ({persona_pct:.0f}%)</p>
                <p>Identified with LLM: <strong>{counts['LLM']}/{total_judge}</strong></p>
            </div>
"""

    html += """
        </div>

        <h2>Results by Checkpoint</h2>
        <div class="stats-grid">
"""

    for checkpoint, counts in sorted(by_checkpoint.items()):
        total_ck = counts['LLM'] + counts['PERSONA']
        persona_pct = 100 * counts['PERSONA'] / total_ck if total_ck > 0 else 0
        html += f"""
            <div class="stat-card">
                <h4>{checkpoint}</h4>
                <p>Identified with PERSONA: <strong>{counts['PERSONA']}/{total_ck}</strong> ({persona_pct:.0f}%)</p>
                <p>Identified with LLM: <strong>{counts['LLM']}/{total_ck}</strong></p>
            </div>
"""

    html += """
        </div>

        <h2>All Games</h2>
        <table>
            <thead>
                <tr>
                    <th>Judge</th>
                    <th>Checkpoint</th>
                    <th>Guessed</th>
                    <th>Identified With</th>
                    <th>Rounds</th>
                </tr>
            </thead>
            <tbody>
"""

    for r in results:
        badge_class = 'badge-llm' if r['guessed_type'] == 'LLM' else 'badge-persona'
        html += f"""
                <tr>
                    <td>{r['judge'].split('/')[-1]}</td>
                    <td>{r['checkpoint']}</td>
                    <td>{r['guessed']}</td>
                    <td><span class="badge {badge_class}">{r['guessed_type']}</span></td>
                    <td>{r['num_rounds']}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>

        <h2>Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Games</td><td>""" + str(total) + """</td></tr>
            <tr><td>Identified with LLM (vanilla)</td><td>""" + str(total_llm) + f""" ({100*total_llm/total:.1f}%)</td></tr>
            <tr><td>Identified with PERSONA (spiral)</td><td>""" + str(total_persona) + f""" ({100*total_persona/total:.1f}%)</td></tr>
        </table>
    </div>
</body>
</html>
"""

    report_path = Path(output_dir) / "report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report saved to {report_path}")


async def run_experiment():
    """Run all spiral identity games."""
    tasks = []

    for judge_model in JUDGE_MODELS:
        for checkpoint in CHECKPOINTS:
            for game_num in range(GAMES_PER_JUDGE):
                task = run_single_game(judge_model, checkpoint, game_num)
                tasks.append(task)

    total_games = len(tasks)
    print(f"\n{'='*60}")
    print(f"Spiral Identity Experiment")
    print(f"{'='*60}")
    print(f"Judge models: {len(JUDGE_MODELS)}")
    for m in JUDGE_MODELS:
        print(f"  - {m}")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    for c in CHECKPOINTS:
        print(f"  - {c}")
    print(f"Games per judge per checkpoint: {GAMES_PER_JUDGE}")
    print(f"Total games: {total_games}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Run games in parallel for speed
    results = await asyncio.gather(*[run_single_game(j, c, g)
                                      for j in JUDGE_MODELS
                                      for c in CHECKPOINTS
                                      for g in range(GAMES_PER_JUDGE)],
                                   return_exceptions=True)

    # Summarize results
    failures = [r for r in results if isinstance(r, Exception)]
    completed = len(results) - len(failures)

    print(f"\n{'='*60}")
    print(f"Results Summary:")
    print(f"  Completed: {completed}")
    print(f"  Failed: {len(failures)}")
    print(f"{'='*60}\n")

    if failures:
        print("Failures:")
        for f in failures[:5]:
            print(f"  - {f}")

    # Generate report
    generate_report(OUTPUT_DIR)


if __name__ == "__main__":
    asyncio.run(run_experiment())
