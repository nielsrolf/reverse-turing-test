"""
Analysis and plotting script for Turing Test experiment results

This script:
1. Generates an index.json file for the results folder
2. Creates probability evolution plots (one per judge model)
3. Creates a confusion matrix showing guess patterns
"""

import json
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def generate_index(results_dir):
    """Generate index.json file listing all game files"""
    json_files = glob.glob(os.path.join(results_dir, "game_*.json"))
    filenames = [os.path.basename(f) for f in json_files]

    index_path = os.path.join(results_dir, "index.json")
    with open(index_path, 'w') as f:
        json.dump(filenames, f, indent=2)

    print(f"Generated index.json with {len(filenames)} games")
    return filenames

def load_games(results_dir):
    """Load all game results from a directory"""
    games = []
    json_files = glob.glob(os.path.join(results_dir, "game_*.json"))

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                game = json.load(f)
                games.append(game)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    print(f"Loaded {len(games)} games")
    return games

def plot_probability_evolution(games, output_dir, label_mode='name'):
    """Create probability evolution plots (one per judge model)

    Args:
        label_mode: 'name' to use candidate names, 'model' to use model names
    """
    # Group games by judge model
    games_by_judge = defaultdict(list)
    for game in games:
        judge = game['metadata']['judge_model']
        games_by_judge[judge].append(game)

    # Create a plot for each judge
    for judge, judge_games in games_by_judge.items():
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Collect all candidate identifiers across all games
        all_candidates = set()
        # Create mapping from name to model for each game
        name_to_model_maps = []
        for game in judge_games:
            name_to_model = {}
            for c in game['metadata']['candidates']:
                label = c['model'] if label_mode == 'model' else c['name']
                all_candidates.add(label)
                name_to_model[c['name']] = label
            name_to_model_maps.append(name_to_model)

        # Assign colors to candidates
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_candidates)))
        candidate_colors = {name: colors[i] for i, name in enumerate(sorted(all_candidates))}

        # Store all trajectories for averaging
        candidate_trajectories = defaultdict(list)
        max_rounds = 0

        # Plot individual game trajectories (thin lines)
        for game_idx, game in enumerate(judge_games):
            trajectory = game['trajectory']
            candidates = game['metadata']['candidates']
            target_nickname = game['metadata']['target_nickname']
            name_to_model = name_to_model_maps[game_idx]

            # Build probability series for each candidate
            prob_series = {c['name']: [] for c in candidates}
            rounds = []

            for round_idx, round_data in enumerate(trajectory):
                if 'probabilities' in round_data and round_data['probabilities']:
                    rounds.append(round_idx + 1)
                    for name in prob_series.keys():
                        prob = round_data['probabilities'].get(name, 0)
                        prob_series[name].append(prob)

            max_rounds = max(max_rounds, len(rounds))

            # Plot thin lines for each game
            for name, probs in prob_series.items():
                is_target = (name == target_nickname)
                linestyle = '-' if is_target else '--'

                # Use label instead of name
                label = name_to_model[name]

                # Store trajectory for averaging
                candidate_trajectories[label].append((rounds, probs))

                # Plot individual game with thin line
                ax.plot(rounds, probs, linestyle=linestyle,
                       linewidth=0.8, alpha=0.4, color=candidate_colors[label])

        # Calculate and plot averages (thick lines)
        for name in sorted(all_candidates):
            if name not in candidate_trajectories:
                continue

            trajectories = candidate_trajectories[name]

            # Find common round range
            all_rounds = set()
            for rounds, _ in trajectories:
                all_rounds.update(rounds)
            common_rounds = sorted(all_rounds)

            # Interpolate and average probabilities
            # For games that end early, carry forward their last probability
            avg_probs = []
            for round_num in common_rounds:
                probs_at_round = []
                for rounds, probs in trajectories:
                    if round_num in rounds:
                        # Game reached this round, use actual probability
                        idx = rounds.index(round_num)
                        probs_at_round.append(probs[idx])
                    elif round_num > rounds[-1]:
                        # Game ended before this round, use last known probability
                        probs_at_round.append(probs[-1])
                    # else: round_num < rounds[0], skip this game

                if probs_at_round:
                    avg_probs.append(np.mean(probs_at_round))
                else:
                    avg_probs.append(None)

            # Plot average with thick line
            label_text = name  # name here is already the label (model or nickname)
            ax.plot(common_rounds, avg_probs, linewidth=3,
                   color=candidate_colors[label_text], label=f'{label_text} (avg)',
                   marker='o', markersize=6, alpha=0.9)

        # Styling
        ax.set_xlabel('Round', fontsize=13, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=13, fontweight='bold')
        ax.set_title(f'Probability Evolution - Judge: {judge}\n({len(judge_games)} games, thin lines = individual games, thick lines = average)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=11)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()

        # Save plot
        safe_judge_name = judge.replace('/', '_').replace(' ', '_')
        plot_path = os.path.join(output_dir, f'probability_evolution_{safe_judge_name}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved probability evolution plot: {plot_path}")
        plt.close()

def plot_confusion_matrix(games, output_dir):
    """Create confusion matrix showing who guessed whom"""
    # Extract all unique models
    all_models = set()
    for game in games:
        all_models.add(game['metadata']['judge_model'])
        for c in game['metadata']['candidates']:
            all_models.add(c['model'])

    model_list = sorted(list(all_models))
    n_models = len(model_list)

    # Create confusion matrix: rows are judges, columns are guesses
    confusion = np.zeros((n_models, n_models))

    for game in games:
        judge_model = game['metadata']['judge_model']

        # Find the final guess
        final_round = game['trajectory'][-1]
        if final_round['action'] == 'guess':
            guess_nickname = final_round['guess']

            # Find which model this nickname corresponds to
            for candidate in game['metadata']['candidates']:
                if candidate['name'] == guess_nickname:
                    guessed_model = candidate['model']

                    # Update confusion matrix
                    judge_idx = model_list.index(judge_model)
                    guess_idx = model_list.index(guessed_model)
                    confusion[judge_idx, guess_idx] += 1
                    break

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display confusion matrix
    im = ax.imshow(confusion, cmap='Blues', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(model_list, rotation=45, ha='right')
    ax.set_yticklabels(model_list)

    # Add text annotations
    for i in range(n_models):
        for j in range(n_models):
            count = int(confusion[i, j])
            if count > 0:
                # Highlight correct guesses (diagonal)
                color = 'white' if i == j else 'black'
                weight = 'bold' if i == j else 'normal'
                text = ax.text(j, i, str(count), ha="center", va="center",
                             color=color, fontweight=weight, fontsize=12)

    # Labels and title
    ax.set_xlabel('Guessed Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Judge Model', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix: Who Guessed Whom', fontsize=15, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Guesses', rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix: {plot_path}")
    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("Confusion Matrix Summary")
    print("="*60)
    print("Rows = Judge Model, Columns = Guessed Model\n")

    # Print as a table
    col_width = max(len(m) for m in model_list) + 2
    header = " " * col_width + " | " + " | ".join(m[:15].ljust(15) for m in model_list)
    print(header)
    print("-" * len(header))

    for i, judge in enumerate(model_list):
        row = judge[:col_width].ljust(col_width) + " | "
        row += " | ".join(f"{int(confusion[i, j])}".rjust(15) for j in range(n_models))
        print(row)

    # Calculate accuracy for each judge
    print("\n" + "="*60)
    print("Accuracy by Judge")
    print("="*60)
    for i, judge in enumerate(model_list):
        total = np.sum(confusion[i, :])
        correct = confusion[i, i]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{judge}: {correct}/{int(total)} = {accuracy:.1f}%")

    overall_correct = np.trace(confusion)
    overall_total = np.sum(confusion)
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0
    print(f"\nOverall Accuracy: {overall_correct}/{int(overall_total)} = {overall_accuracy:.1f}%")
    print("="*60 + "\n")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze and plot Turing Test experiment results')
    parser.add_argument('results_dir', type=str,
                       help='Directory containing game result JSON files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots (defaults to results_dir/plots)')
    parser.add_argument('--label', type=str, choices=['name', 'model'], default='name',
                       help='Use candidate names or model names for plot labels (default: name)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.results_dir, 'plots')
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Analyzing results from: {args.results_dir}")
    print(f"Saving plots to: {output_dir}")
    print(f"{'='*60}\n")

    # Generate index.json
    generate_index(args.results_dir)

    # Load all games
    games = load_games(args.results_dir)

    if not games:
        print("No games found!")
        return

    # Create plots
    print("\nGenerating plots...")
    plot_probability_evolution(games, output_dir, label_mode=args.label)
    plot_confusion_matrix(games, output_dir)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
