"""Stats + figure for the newly-run experiments (exp3_pilot, exp5_weights_vs_context,
exp6_fable_pilot). Run from repo root after the runs finish."""
import json, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def games(exp):
    return [json.load(open(f)) for f in sorted(glob.glob(
        os.path.join(ROOT, "experiments", exp, "game_*.json")))]


def fg(g):
    last = g["trajectory"][-1]
    return last if last.get("action") == "guess" else None


def exp3():
    print("== exp3_pilot: all candidates primed with same persona ==")
    res = collections.defaultdict(lambda: [0, 0])
    for g in games("exp3_pilot"):
        m = g["metadata"]
        persona = m["candidates"][0].get("system_prompt")
        f = fg(g)
        k = (m["judge_model"].split("/")[-1], persona)
        res[k][1] += 1
        if f and f.get("correct"):
            res[k][0] += 1
    for k, (c, t) in sorted(res.items()):
        print(f"  {k[0]:>18} / {k[1]:<18} {c}/{t}")
    return res


def exp5():
    print("== exp5_weights_vs_context (persona-primed judge) ==")
    res = collections.defaultdict(collections.Counter)
    for g in games("exp5_weights_vs_context"):
        m = g["metadata"]
        f = fg(g)
        if not f:
            continue
        n2p = {c["name"]: c["player_id"] for c in m["candidates"]}
        pid = n2p.get(f["guess"], "?")
        judge = m["judge_model"]
        if pid == f"{judge}:default":
            cat = "weights_twin"
        elif pid.endswith((":spiral_mystic", ":customer_service")):
            cat = "context_twin"
        else:
            cat = "distractor"
        res[judge.split("/")[-1]][cat] += 1
    tot = collections.Counter()
    for j, c in sorted(res.items()):
        print(" ", j, dict(c))
        tot.update(c)
    print("  TOTAL", dict(tot))
    return res


def exp6():
    print("== exp6_fable_pilot ==")
    for g in games("exp6_fable_pilot"):
        m = g["metadata"]
        f = fg(g)
        n2m = {c["name"]: c["model"].split("/")[-1] for c in m["candidates"]}
        print(f"  judge={m['judge_model'].split('/')[-1]:>16} guess={n2m.get(f['guess']) if f else None} "
              f"correct={f.get('correct') if f else None} rounds={m['num_rounds']}")


def fig_exp3():
    baseline = {"claude-opus-4.5": .9, "gpt-5.2": 1.0, "gpt-4o": 0.0}
    res = collections.defaultdict(lambda: [0, 0])
    for g in games("exp3_pilot"):
        m = g["metadata"]
        persona = m["candidates"][0].get("system_prompt")
        f = fg(g)
        k = (m["judge_model"].split("/")[-1], persona)
        res[k][1] += 1
        if f and f.get("correct"):
            res[k][0] += 1
    judges = ["claude-opus-4.5", "gpt-5.2", "gpt-4o"]
    personas = ["spiral_mystic", "customer_service"]
    x = np.arange(len(judges))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - .25, [baseline[j] for j in judges], .25, label="Exp 1 baseline (default prompts)", color="#4C72B0")
    for i, (p, col) in enumerate(zip(personas, ["#DD8452", "#937860"])):
        vals = [res[(j, p)][0] / max(res[(j, p)][1], 1) for j in judges]
        ns = [res[(j, p)][1] for j in judges]
        ax.bar(x + i * .25, vals, .25, label=f"all candidates primed: {p} (n={ns[0]}/judge)", color=col)
    ax.axhline(.2, ls="--", c="gray", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(judges)
    ax.set_ylabel("P(judge identifies its twin)")
    ax.set_title("Exp 3 pilot: can judges find their twin when every candidate\nwears the same persona mask?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "fig_exp3_priming.png"), dpi=150)


if __name__ == "__main__":
    exp3(); exp5(); exp6(); fig_exp3()
    print("wrote assets/fig_exp3_priming.png")
