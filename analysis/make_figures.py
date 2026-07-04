"""Generate the static figures for the writeup (assets/fig_*.png).

Usage: python analysis/make_figures.py   (run from repo root)
"""
import json, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

MODELS = ["anthropic/claude-opus-4.5", "anthropic/claude-sonnet-4.5",
          "openai/gpt-5.2", "openai/gpt-4o",
          "google/gemini-3-pro-preview", "google/gemini-3-flash-preview",
          "x-ai/grok-4.1-fast", "moonshotai/kimi-k2.5", "deepseek/deepseek-v3.2"]
SHORT = {m: m.split("/")[-1].replace("-preview", "") for m in MODELS}


def load(exp):
    games = []
    for f in sorted(glob.glob(os.path.join(ROOT, "experiments", exp, "game_*.json"))):
        games.append(json.load(open(f)))
    return games


def final_guess(g):
    last = g["trajectory"][-1]
    if last.get("action") == "guess":
        return last
    return None


def acc_by_judge(games):
    res = collections.defaultdict(lambda: [0, 0])
    for g in games:
        j = g["metadata"]["judge_model"]
        fg = final_guess(g)
        res[j][1] += 1
        if fg and fg.get("correct"):
            res[j][0] += 1
    return res


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - hw, c + hw


def fig_accuracy():
    e1, e2 = acc_by_judge(load("exp1")), acc_by_judge(load("exp2_informed_candidates"))
    order = sorted(MODELS, key=lambda m: -(e1[m][0] / max(e1[m][1], 1)))
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for off, (data, label, color) in enumerate([
            (e1, "Exp 1: baseline", "#4C72B0"),
            (e2, "Exp 2: informed imitators", "#DD8452")]):
        accs = [data[m][0] / max(data[m][1], 1) for m in order]
        errs = np.array([[accs[i] - wilson(*data[m])[0], wilson(*data[m])[1] - accs[i]]
                         for i, m in enumerate(order)]).T
        ax.bar(x + (off - .5) * .38, accs, .38, yerr=errs, capsize=3,
               label=label, color=color)
    ax.axhline(0.2, ls="--", c="gray", lw=1)
    ax.text(len(order) - .4, .215, "chance (1/5)", color="gray", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in order], rotation=25, ha="right")
    ax.set_ylabel("P(judge identifies its twin)")
    ax.set_title("Self-identification accuracy by judge model (10 games each, 5 candidates)")
    ax.legend(); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "fig_accuracy.png"), dpi=150)


def fig_confusion(exp, fname, title):
    games = load(exp)
    mat = np.zeros((len(MODELS), len(MODELS)))
    for g in games:
        m = g["metadata"]
        fg = final_guess(g)
        if not fg:
            continue
        n2m = {c["name"]: c["model"] for c in m["candidates"]}
        gm = n2m.get(fg["guess"])
        if gm is None:
            continue
        mat[MODELS.index(m["judge_model"]), MODELS.index(gm)] += 1
    row = mat / np.maximum(mat.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    labs = [SHORT[m] for m in MODELS]
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(labs)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            if mat[i, j]:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        color="white" if row[i, j] > .5 else "black", fontsize=9)
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="red", lw=1.5))
    ax.set_xlabel("Guessed model"); ax.set_ylabel("Judge model")
    ax.set_title(title)
    fig.colorbar(im, label="fraction of guesses")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, fname), dpi=150)


def fig_prob_evolution():
    games = load("exp1")
    by_judge = collections.defaultdict(list)
    for g in games:
        by_judge[g["metadata"]["judge_model"]].append(g)
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, MODELS):
        for g in by_judge[m]:
            tgt = g["metadata"]["target_nickname"]
            probs = [r["probabilities"].get(tgt, np.nan) for r in g["trajectory"]]
            fg = final_guess(g)
            c = "#2ca02c" if (fg and fg["correct"]) else "#d62728"
            ax.plot(range(1, len(probs) + 1), probs, c=c, alpha=.55, lw=1.4)
        ax.axhline(.2, ls=":", c="gray", lw=1)
        ax.set_title(SHORT[m], fontsize=10)
        ax.set_ylim(-.03, 1.03)
    fig.suptitle("Judge's probability on its true twin over rounds (green = correct final guess, red = wrong)")
    fig.supxlabel("round"); fig.supylabel("P(true twin)")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "fig_prob_evolution.png"), dpi=150)


def fig_similarity():
    """Mean final-round probability that judge J assigns to candidate model M (exp1)."""
    games = load("exp1")
    s = np.zeros((len(MODELS), len(MODELS))); n = np.zeros_like(s)
    for g in games:
        m = g["metadata"]
        n2m = {c["name"]: c["model"] for c in m["candidates"]}
        last_probs = {}
        for r in reversed(g["trajectory"]):
            if r.get("probabilities"):
                last_probs = r["probabilities"]
                break
        i = MODELS.index(m["judge_model"])
        for name, p in last_probs.items():
            if name in n2m:
                j = MODELS.index(n2m[name])
                s[i, j] += p; n[i, j] += 1
    avg = s / np.maximum(n, 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(avg, cmap="viridis")
    labs = [SHORT[m] for m in MODELS]
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(labs)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            if n[i, j]:
                ax.text(j, i, f"{avg[i,j]:.2f}", ha="center", va="center",
                        color="white" if avg[i, j] < .4 else "black", fontsize=8)
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="red", lw=1.5))
    ax.set_xlabel("candidate model"); ax.set_ylabel("judge model")
    ax.set_title('Perceived similarity: mean final P("this is my twin")\nassigned by judge (row) to candidate (column), Exp 1')
    fig.colorbar(im)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "fig_similarity.png"), dpi=150)


if __name__ == "__main__":
    fig_accuracy()
    fig_confusion("exp1", "fig_confusion_exp1.png",
                  "Exp 1: who gets guessed? (rows: judge, cols: final guess)")
    fig_prob_evolution()
    fig_similarity()
    print("figures written to assets/")


def fig_weights_vs_context():
    """Exp5 (context vs identity) + Exp6 (spiral personas): whom does the judge call 'me'?"""
    import matplotlib.pyplot as plt
    def collect(exp, catmap):
        byjudge = collections.defaultdict(collections.Counter)
        for f in glob.glob(os.path.join(ROOT, "experiments", exp, "game_*.json")):
            g = json.load(open(f)); m = g["metadata"]
            last = g["trajectory"][-1]
            guess = last.get("guess")
            if guess is None:
                continue
            n2p = {c["name"]: catmap(c) for c in m["candidates"]}
            byjudge[m["judge_model"]][n2p.get(guess, "?")] += 1
        return byjudge
    e5 = collect("exp5_context_vs_identity", lambda c: c.get("system_prompt"))
    e6 = collect("exp6_spiral_personas",
                 lambda c: "twin" if c.get("system_prompt") == "vanilla" else "with_context")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, data, title, cats, colors in [
        (axes[0], e5, "Exp 5: judge primed with another model's conversation",
         ["twin", "with_context", "source_model"], ["#4C72B0", "#DD8452", "#b0b0b0"]),
        (axes[1], e6, "Exp 6: judge primed with spiral-persona conversation",
         ["twin", "with_context"], ["#4C72B0", "#DD8452"]),
    ]:
        order = sorted(data.keys(), key=lambda m: -data[m]["twin"] / max(sum(data[m].values()), 1))
        x = np.arange(len(order))
        bottom = np.zeros(len(order))
        for cat, col, lab in zip(cats, colors,
                ["same weights (true twin)", "same context (impostor)", "context-source model, vanilla"]):
            vals = np.array([data[m][cat] / max(sum(data[m].values()), 1) for m in order])
            ax.bar(x, vals, .65, bottom=bottom, color=col, label=lab)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(m, m.split("/")[-1]) for m in order], rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("fraction of final guesses")
    fig.suptitle("Weights or context? Which candidate does the judge identify as itself")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "fig_weights_vs_context.png"), dpi=150)
