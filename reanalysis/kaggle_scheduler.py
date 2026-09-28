"""Push queued Kaggle kernels (max N concurrent GPU sessions), poll status, download outputs when complete.
Usage: python kaggle_scheduler.py queue.json   (state kept in kaggle_runs/state.json; safe to re-run)"""
import os as _os  # (release) all paths below are relative to ROOT (repository root by default; see README)
ROOT = (_os.environ.get("PG26038_ROOT") or _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))).replace(chr(92), "/")
import json, os, subprocess, sys, time

K = _os.environ.get("KAGGLE_CLI", "kaggle")  # (release) Kaggle CLI executable
RUNS = (ROOT + "/reanalysis/kaggle_runs")
MAXC = 2; POLL = 240
queue = json.load(open(os.path.join(RUNS, sys.argv[1] if len(sys.argv) > 1 else "queue.json")))
qname = os.path.splitext(sys.argv[1] if len(sys.argv) > 1 else "queue.json")[0]
state_path = os.path.join(RUNS, "state.json" if qname in ("queue", "queue_mb") else f"state_{qname}.json")
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
for s in queue: state.setdefault(s, "pending")

def save(): json.dump(state, open(state_path, "w"), indent=1)
def run(args):
    p = subprocess.run([K] + args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (p.stdout + p.stderr).strip()

def status(slug):
    out = run(["kernels", "status", f"carlychery2001/{slug}"]).lower()
    for s in ("complete", "error", "cancel", "running", "queued"):
        if s in out: return s, out
    return "unknown", out

t0 = time.time()
while True:
    for slug in [s for s in queue if state[s] in ("running", "queued", "pushed")]:
        st, raw = status(slug)
        if st == "complete":
            d = os.path.join(RUNS, slug, "output"); os.makedirs(d, exist_ok=True)
            print(f"[{(time.time()-t0)/60:.0f}m] {slug} complete -> downloading", flush=True)
            print(run(["kernels", "output", f"carlychery2001/{slug}", "-p", d])[-300:], flush=True)
            state[slug] = "done"
        elif st in ("error", "cancel"):
            d = os.path.join(RUNS, slug, "output"); os.makedirs(d, exist_ok=True)
            run(["kernels", "output", f"carlychery2001/{slug}", "-p", d])
            print(f"[{(time.time()-t0)/60:.0f}m] {slug} FAILED: {raw[-200:]}", flush=True); state[slug] = "failed"
        elif st in ("running", "queued"):
            state[slug] = st
    active = sum(state[s] in ("running", "queued", "pushed") for s in queue)
    for slug in [s for s in queue if state[s] == "pending"][:max(0, MAXC - active)]:
        out = run(["kernels", "push", "-p", os.path.join(RUNS, slug)])
        ok = "successfully pushed" in out.lower() or "kernel version" in out.lower()
        print(f"[{(time.time()-t0)/60:.0f}m] push {slug}: {'OK' if ok else out[-250:]}", flush=True)
        if ok: state[slug] = "pushed"
    save()
    left = [s for s in queue if state[s] not in ("done", "failed")]
    print(f"[{(time.time()-t0)/60:.0f}m] status: " + ", ".join(f"{s.replace('pg26038-', '')}={state[s]}" for s in queue), flush=True)
    if not left:
        print("ALL KERNELS FINISHED", flush=True); break
    time.sleep(POLL)
