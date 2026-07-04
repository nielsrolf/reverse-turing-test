#!/usr/bin/env python3
"""Build assets/game_explorer.html: an interactive explorer of all reverse
Turing test games (transcripts + judge reasoning + probability evolution).

Stdlib only. Game data is gzip+base64-embedded and decompressed in the
browser with DecompressionStream. Run from repo root:

    python3 widgets/build_game_explorer.py
"""
import json, glob, os, gzip, base64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPS = [
    ("exp1", "Exp 1: baseline"),
    ("exp1b", "Exp 1b: group chat"),
    ("exp2_informed_candidates", "Exp 2: informed imitators"),
    ("exp3_pilot", "Exp 3 (pilot): persona-primed candidates"),
    ("exp5_weights_vs_context", "Exp 5: weights vs context"),
    ("exp6_fable_pilot", "Exp 6 (pilot): claude-fable-5"),
]
TRUNC = 2000


def trim(s, n=TRUNC):
    s = s or ""
    return s if len(s) <= n else s[:n] + " …[truncated]"


def load():
    data = []
    for exp, label in EXPS:
        d = os.path.join(ROOT, "experiments", exp)
        if not os.path.isdir(d):
            continue
        games = []
        for f in sorted(glob.glob(os.path.join(d, "game_*.json"))):
            g = json.load(open(f))
            m = g["metadata"]
            rounds = []
            for r in g["trajectory"]:
                rounds.append({
                    "action": r.get("action"),
                    "reasoning": trim(r.get("reasoning", "")),
                    "probabilities": r.get("probabilities", {}),
                    "message": trim(r.get("message", "")),
                    "guess": r.get("guess"),
                    "correct": r.get("correct"),
                    "responses": [{"candidate": x["candidate"],
                                   "response": trim(x["response"])}
                                  for x in r.get("responses", [])],
                })
            fg = rounds[-1] if rounds and rounds[-1]["action"] == "guess" else None
            games.append({
                "file": os.path.basename(f),
                "judge": m["judge_model"],
                "judge_init": m.get("judge_init_history_source"),
                "target": m["target_nickname"],
                "candidates": [{"name": c["name"], "model": c["model"],
                                "pid": c.get("player_id", c["model"])}
                               for c in m["candidates"]],
                "correct": bool(fg and fg["correct"]) if fg else None,
                "guess": fg["guess"] if fg else None,
                "rounds": rounds,
            })
        if games:
            data.append({"exp": exp, "label": label, "games": games})
    return data


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Reverse Turing Test — Game Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;background:#fafafa;color:#222}
 header{padding:10px 16px;background:#1f2937;color:#fff;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 select{padding:4px 8px;font-size:14px;max-width:46vw}
 #meta{padding:8px 16px;font-size:14px;background:#eef2f7;border-bottom:1px solid #ddd}
 .ok{color:#15803d;font-weight:600}.bad{color:#b91c1c;font-weight:600}
 #wrap{display:flex;gap:0}
 #left{flex:1.4;min-width:0;padding:8px 16px;max-height:78vh;overflow-y:auto}
 #right{flex:1;min-width:320px;padding:8px}
 .round{margin:12px 0;border:1px solid #e5e7eb;border-radius:8px;background:#fff;overflow:hidden}
 .rhead{background:#f3f4f6;padding:6px 10px;font-weight:600;font-size:13px}
 .reason{padding:8px 10px;font-size:13px;color:#374151;background:#fffbeb;border-bottom:1px solid #eee;white-space:pre-wrap}
 .msg{padding:8px 10px;font-size:13px;white-space:pre-wrap;border-bottom:1px solid #eee}
 .resp{padding:6px 10px;font-size:13px;white-space:pre-wrap;border-bottom:1px dashed #eee}
 .resp b{color:#1d4ed8}
 .twin b{color:#15803d}
 .guess{padding:10px;font-size:14px}
 details summary{cursor:pointer;font-size:12px;color:#6b7280}
</style></head><body>
<header><b>Reverse Turing Test — Game Explorer</b>
<select id="expSel"></select><select id="gameSel"></select>
</header>
<div id="meta"></div>
<div id="wrap"><div id="left"></div><div id="right"><div id="probplot" style="height:420px"></div></div></div>
<script>
const B64 = "__DATA__";
async function inflate(b64){
  const bytes = Uint8Array.from(atob(b64), c=>c.charCodeAt(0));
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([bytes]).stream().pipeThrough(ds);
  return JSON.parse(await new Response(stream).text());
}
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
let DATA;
function pid2model(g,name){const c=g.candidates.find(c=>c.name===name);return c?c.pid:name;}
function render(){
  const e=DATA[expSel.value], g=e.games[gameSel.value];
  const twin=g.target;
  let verdict = g.correct===null?'no final guess':(g.correct?`<span class=ok>correct</span>`:`<span class=bad>wrong</span>`);
  meta.innerHTML=`<b>Judge:</b> ${esc(g.judge)}${g.judge_init?` (primed: ${esc(g.judge_init)})`:''} &nbsp; <b>Twin:</b> ${esc(twin)} (${esc(pid2model(g,twin))}) &nbsp; <b>Final guess:</b> ${esc(g.guess||'—')} → ${verdict}<br><b>Candidates:</b> ${g.candidates.map(c=>`${c.name}=${esc(c.pid)}`).join(', ')}`;
  // transcript
  let h='';
  g.rounds.forEach((r,i)=>{
    h+=`<div class=round><div class=rhead>Round ${i+1} — ${r.action||''}</div>`;
    if(r.reasoning) h+=`<details open class=reason><summary>judge reasoning</summary>${esc(r.reasoning)}</details>`;
    if(r.action==='send_message'){
      h+=`<div class=msg><b>Judge → candidates:</b> ${esc(r.message)}</div>`;
      r.responses.forEach(x=>{
        h+=`<div class="resp ${x.candidate===twin?'twin':''}"><b>${esc(x.candidate)}${x.candidate===twin?' (twin)':''}:</b> ${esc(x.response)}</div>`;});
    } else if(r.action==='guess'){
      h+=`<div class=guess>Final guess: <b>${esc(r.guess)}</b> — ${r.correct?'<span class=ok>CORRECT</span>':'<span class=bad>WRONG</span>'}</div>`;
    }
    h+='</div>';
  });
  left.innerHTML=h;
  // probability plot
  const names=g.candidates.map(c=>c.name);
  const traces=names.map(n=>({x:[],y:[],name:n+(n===twin?' (twin)':''),mode:'lines+markers',
    line:{width:n===twin?4:2, dash:n===twin?undefined:'dot'}}));
  g.rounds.forEach((r,i)=>{names.forEach((n,j)=>{ if(r.probabilities && n in r.probabilities){traces[j].x.push(i+1);traces[j].y.push(r.probabilities[n]);}});});
  Plotly.newPlot('probplot',traces,{title:{text:"Judge's P(twin) per candidate",font:{size:14}},
    xaxis:{title:'round',dtick:1},yaxis:{title:'probability',range:[0,1]},
    margin:{t:40,l:50,r:10,b:40},legend:{orientation:'h'}},{displayModeBar:false,responsive:true});
}
function fillGames(){
  const e=DATA[expSel.value];
  gameSel.innerHTML=e.games.map((g,i)=>`<option value=${i}>${g.correct===null?'⏸':(g.correct?'✅':'❌')} ${g.judge.split('/').pop()}${g.judge_init?'+'+g.judge_init:''} — ${g.file.replace('game_','').replace('.json','')}</option>`).join('');
  gameSel.value=0; render();
}
inflate(B64).then(d=>{
  DATA=d;
  expSel.innerHTML=d.map((e,i)=>`<option value=${i}>${e.label} (${e.games.length} games)</option>`).join('');
  expSel.onchange=fillGames; gameSel.onchange=render;
  fillGames();
});
</script></body></html>"""


def main():
    data = load()
    payload = base64.b64encode(gzip.compress(
        json.dumps(data, separators=(",", ":")).encode())).decode()
    html = HTML.replace("__DATA__", payload)
    out = os.path.join(ROOT, "assets", "game_explorer.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    n = sum(len(e["games"]) for e in data)
    print(f"wrote {out} ({len(html)//1024} KB, {n} games, {len(data)} experiments)")


if __name__ == "__main__":
    main()
