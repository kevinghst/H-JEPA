"""SLURM launcher for H-JEPA training, resume and per-epoch planning eval, any env. Run from h_jepa/.

  python launch.py train --config-name cube_lewm --sweep crop_ab [--seeds 42,43,44] \\
      [--grid level1.wm.history_size=3,7 ...] [--into <sweep_dir>] [hydra overrides ...]
  python launch.py resume <run_dir> [hydra overrides, e.g. trainer.max_epochs=100]
  python launch.py eval <sweep_dir|run_dir> [--epochs all|last|N,M] [eval.py overrides]
  common flags: --partition P --account A --qos Q --time HH:MM:SS --gpus N --mem 200G --dry

train   One job per (grid cell x seed) in $HJEPA_HOME/ckpts/<env>/<sweep>_<YYYY-MM-DD_HH-MM>/<cell>/seed<S>,
        <cell> = output_model_name = one token per grid key (level2.wm.history_size=7 -> l2hs7). The tree
        (git ls-files -co --exclude-standard) is copied once to <sweep_dir>/code (+ GIT_COMMIT, GIT_STATUS,
        UNCOMMITTED.diff); every job of the sweep, its resumes and --into additions run from that copy.
resume  Relaunch one run from its saved config.yaml; training restarts from lightning_resume/last.ckpt.
eval    One job per <run>/<name>_epoch_<N>_object.ckpt under the target: eval.py --config-name <env>_{flat|l<n>},
        planner seed = model seed -> <run>/eval_epoch/epoch_<N>/ (metrics.yaml; DROID: eval.csv). Evals run from
        <sweep_dir>/eval_code, frozen from the worktree at the sweep's first eval (mv it aside to refresh).
Every check (config composes, datasets exist, no live job, fresh dirs absent) runs before any sbatch.
Cluster settings: config/slurm/default.yaml < local.yaml < flags. Jobs requeue on preemption, append to
<run_dir>/slurm/%x_%j.out and run the code copy's main_hjepa.py (srun, one task per GPU) or eval.py.
"""

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

H = Path(__file__).resolve().parent
REPO = H.parent
SET_BY_LAUNCHER = {"seed", "subdir", "output_model_name", "trainer.devices"}
LIVE = "PENDING,CONFIGURING,RUNNING,SUSPENDED,REQUEUED,REQUEUE_HOLD,RESIZING"  # not COMPLETING
EPOCH_RE = re.compile(r"_epoch_(\d+)_object\.ckpt$")


def die(msg: str) -> None:
    sys.exit(f"launch.py: ERROR: {msg}")


def hjepa_home() -> Path:
    h = os.environ.get("HJEPA_HOME")
    if not h or not Path(h).is_dir():
        die(f"HJEPA_HOME must be set to an existing dir (got {h!r})")
    return Path(h).resolve()


def cluster(sub: str, a: argparse.Namespace) -> OmegaConf:
    c = OmegaConf.load(H / "config/slurm/default.yaml")
    if (H / "config/slurm/local.yaml").exists():
        c = OmegaConf.merge(c, OmegaConf.load(H / "config/slurm/local.yaml"))
    if a.partition and not a.account:
        die("--partition needs --account too (they are set together)")
    flags = {k: getattr(a, k) for k in ("partition", "account", "qos", "time", "mem") if getattr(a, k)}
    c = OmegaConf.merge(c, c.get(sub) or {}, flags, {"gpus_per_node": a.gpus} if a.gpus else {})
    keys = ("venv", "partition", "account", "qos", "gpus_per_node", "cpus_per_task", "mem", "time")
    if missing := [k for k in keys if c.get(k) in (None, "")]:
        die(f"cluster settings missing {missing}: set them in {H}/config/slurm/local.yaml")
    if not Path(c.venv, "bin/python").exists():
        die(f"venv {c.venv} has no bin/python")
    return c


def compose_cfg(config_dir: Path, name: str, overrides: list) -> OmegaConf:
    GlobalHydra.instance().clear()
    ov = [o for o in overrides if not o.lstrip("+~").startswith("hydra.")]
    try:
        with initialize_config_dir(str(config_dir), version_base=None):
            return compose(name, overrides=ov)
    except Exception as e:
        die(f"config {config_dir}/{name} does not compose with {ov}:\n  {type(e).__name__}: {e}")


def check_datasets(cfg: OmegaConf, home: Path) -> None:
    for k in ("name", "val_name"):
        if v := OmegaConf.select(cfg, f"data.dataset.{k}", default=None):
            p = home / "droid" / v if v.endswith(".csv") else home / f"{v}.h5"
            p.exists() or die(f"data.dataset.{k}={v}: {p} not found")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def snapshot(dest: Path) -> None:
    dest.mkdir(parents=True)
    files = git("ls-files", "-co", "--exclude-standard")
    rsync = ["rsync", "-a", "--ignore-missing-args", "--files-from=-", ".", str(dest)]
    subprocess.run(rsync, cwd=REPO, input=files, text=True, check=True)
    if os.path.lexists(H / "assets") and not os.path.lexists(dest / "h_jepa/assets"):
        (dest / "h_jepa/assets").symlink_to(os.path.realpath(H / "assets"))  # gitignored eval tasks
    for f, args in (("GIT_COMMIT", ["rev-parse", "HEAD"]), ("GIT_STATUS", ["status", "--short"]),
                    ("UNCOMMITTED.diff", ["diff", "HEAD"])):  # fmt: skip
        (dest / f).write_text(git(*args))
    print(f"code snapshot {dest} @ {git('rev-parse', '--short', 'HEAD').strip()}")


def find_snapshot(p: Path, name: str = "code") -> Path | None:  # the sweep's <name> code copy above p
    code = next((d / name for d in [p, *p.parents] if (d / name / "GIT_COMMIT").exists()), None)
    head = [git("rev-parse", "HEAD"), git("diff", "HEAD")]
    if code and [(code / f).read_text() for f in ("GIT_COMMIT", "UNCOMMITTED.diff")] != head:
        print(f"WARNING: {code} differs from the worktree; jobs run it, not your edits (mv it aside to refresh)")
    return code


def epoch_ckpts(d: Path) -> dict:
    return {int(m.group(1)): c for c in d.glob("*_epoch_*_object.ckpt") if (m := EPOCH_RE.search(c.name))}


def is_run(d: Path) -> bool:
    return (d / "config.yaml").exists() and bool(epoch_ckpts(d) or (d / "lightning_resume").exists())


def submit(c, home: Path, name: str, log_dir: Path, gpus: int, code: Path, args: list, dry: bool) -> str | None:
    job = shlex.join([f"{c.venv}/bin/python", str(H / "launch.py"), "_job", str(code), *args])
    cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={c.partition}",
           f"--account={c.account}", f"--qos={c.qos}", f"--time={c.time}", "--nodes=1", "--requeue",
           "--open-mode=append", f"--ntasks-per-node={gpus}", f"--gpus-per-node={gpus}", f"--mem={c.mem}",
           f"--cpus-per-task={c.cpus_per_task}", *(f"--{k}={log_dir}/%x_%j.out" for k in ("output", "error")),
           f"--export=ALL,HJEPA_HOME={home}", f"--wrap={job}"]  # fmt: skip
    if dry:
        print("DRY", shlex.join(cmd))
        return "DRY"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}  # nested-submit leak
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    jid = r.stdout.strip().split(";")[0]
    if r.returncode or not jid.isdigit():
        print(f"SUBMIT FAILED {name}: rc={r.returncode} {r.stderr.strip()}")
        return None
    print(f"submitted {jid} {name}")
    return jid


def done_file(config: str, out: Path) -> Path:
    return out / ("eval.csv" if config.startswith("droid") else "metrics.yaml")


def run_job(code: str, mode: str, args: list) -> None:
    """Job body: the code copy's `train <main_hjepa.py args>` or `eval <config|seed|ckpt|outdir>... -- <overrides>`."""
    os.chdir(f"{code}/h_jepa")
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    os.environ.update(PYTHONPATH=f"{code}:{code}/h_jepa", PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
                      PYTHONDONTWRITEBYTECODE="1", HYDRA_FULL_ERROR="1", PYTHONFAULTHANDLER="1", MUJOCO_GL="egl",
                      PYOPENGL_PLATFORM="egl", MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      WANDB_MODE="disabled" if mode == "eval" else os.environ.get("WANDB_MODE", "online"))
    jid = os.environ["SLURM_JOB_ID"]
    print(f"{mode} code={code} job={jid} restart={os.environ.get('SLURM_RESTART_COUNT', 0)}", flush=True)
    if mode == "train":  # exec: srun gets the job's signals (preemption -> spt requeue handler)
        os.environ["MASTER_PORT"] = str(20000 + int(jid) % 20000)
        os.execvp("srun", ["srun", sys.executable, "main_hjepa.py", *args])
    specs, ov, rc = args[: args.index("--")], args[args.index("--") + 1 :], 0
    for config, seed, ckpt, out in (spec.split("|") for spec in specs):
        if not done_file(config, Path(out)).exists():
            rc |= subprocess.run([sys.executable, "eval.py", "--config-name", config, f"seed={seed}", f"policy={ckpt}",
                                  f"output.dir={out}", f"hydra.run.dir={out}/hydra", *ov]).returncode != 0  # fmt: skip
    sys.exit(rc)


def record(d: Path, entry: dict) -> None:
    f = d / "launch.json"
    log = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps([*log, {"time": f"{datetime.now():%F %T}", "argv": sys.argv, **entry}], indent=1))


def key_token(key: str) -> str:
    parts = key.lstrip("+~").split(".")
    lvl = "".join(f"l{m.group(1)}" for p in parts if (m := re.fullmatch(r"level(\d+)", p)))
    return lvl + "".join(w[0] for w in parts[-1].split("_") if w)


def safe(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "-", v).strip("-") or "x"


def cmd_train(a, overrides: list) -> None:
    home, c = hjepa_home(), cluster("train", a)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.sweep):
        die(f"--sweep {a.sweep!r}: give a descriptive, filesystem-safe name")
    for o in overrides:
        if (k := o.split("=")[0].lstrip("+~")) in SET_BY_LAUNCHER or k.startswith("hydra."):
            die(f"{k} is set by launch.py")
    grid = [(k, vs.split(",")) for k, _, vs in (g.partition("=") for g in a.grid)]
    if bad := [k for k, vs in grid if vs == [""] or k.lstrip("+~") in SET_BY_LAUNCHER]:
        die(f"--grid {bad}: expected key=v1,v2 on keys launch.py does not set")
    toks = [key_token(k) for k, _ in grid]
    if len(set(toks)) < len(toks):
        toks = [safe(k.lstrip("+~")) + "-" for k, _ in grid]
    if a.into:
        sweep_dir = Path(a.into).resolve()
        if not (sweep_dir / "code/GIT_COMMIT").exists():
            die(f"--into {sweep_dir}: not a launch.py sweep dir (no code/GIT_COMMIT)")
    else:
        env = compose_cfg(H / "config/train", a.config_name, overrides).env
        sweep_dir = home / "ckpts" / env / f"{a.sweep}_{datetime.now():%Y-%m-%d_%H-%M}"
        if sweep_dir.exists():
            die(f"{sweep_dir} exists (launched this minute already?): wait a minute or use --into")
    rel = sweep_dir.relative_to(home / "ckpts")
    jobs = []
    for combo in itertools.product(*[vs for _, vs in grid]):
        cell = "_".join(t + safe(v) for t, v in zip(toks, combo)) or a.config_name
        for s in [int(s) for s in a.seeds.split(",")]:
            run_dir = sweep_dir / cell / f"seed{s}"
            ov = [*overrides, *(f"{k}={v}" for (k, _), v in zip(grid, combo)), f"seed={s}",
                  f"subdir={rel}/{cell}/seed{s}", f"output_model_name={cell}",
                  f"trainer.devices={c.gpus_per_node}"]  # fmt: skip
            if run_dir.exists():
                die(f"{run_dir} exists: a fresh train would silently resume it; use `resume` or a new sweep")
            cfg = compose_cfg(H / "config/train", a.config_name, ov)
            check_datasets(cfg, home)
            if cfg.wandb.enabled:
                ov += [f"++wandb.config.group={a.sweep}", f"++wandb.config.name={cell}_s{s}"]
            jobs.append((cell, s, run_dir, [*ov, f"hydra.run.dir={run_dir}/hydra"]))
    print(f"sweep {sweep_dir}: {len(jobs)} jobs ({len(jobs) // len(a.seeds.split(','))} cells x {a.seeds})")
    code = sweep_dir / "code"
    if a.into:
        find_snapshot(sweep_dir)
    elif not a.dry:
        snapshot(code)
    ids = {}
    for cell, s, run_dir, ov in jobs:
        args = ["train", "--config-name", a.config_name, *ov]
        ids[str(run_dir)] = submit(c, home, f"hj_{a.sweep}_{cell}_s{s}", run_dir / "slurm", c.gpus_per_node, code,
                                   args, a.dry)  # fmt: skip
    if not a.dry:
        commit = (code / "GIT_COMMIT").read_text().strip()
        record(sweep_dir, {"cmd": "train", "config_name": a.config_name, "overrides": overrides, "grid": dict(grid),
                           "seeds": a.seeds, "jobs": ids, "snapshot_commit": commit})


def cmd_resume(a, overrides: list) -> None:
    home, c = hjepa_home(), cluster("resume", a)
    rd = Path(a.run_dir).resolve()
    if below := [d for d in [*rd.glob("*"), *rd.glob("*/*")] if d.is_dir() and is_run(d)]:
        die(f"{rd} contains {len(below)} run dirs (a sweep dir?): resume one run dir")
    if not (rd / "lightning_resume/last.ckpt").exists() or not (rd / "config.yaml").exists():
        die(f"nothing to resume in {rd}: needs config.yaml and lightning_resume/last.ckpt (epoch ckpts are not)")
    saved = OmegaConf.load(rd / "config.yaml")
    for f in (rd / "launch.json", rd.parent.parent / "launch.json"):
        log = json.loads(f.read_text()) if f.exists() else []
        jid = next((e["jobs"][str(rd)] for e in reversed(log) if str(rd) in e.get("jobs", {})), None)
        sq = ["squeue", "-h", "-j", str(jid), "-t", LIVE, "-o", "%T"]
        if jid and (st := subprocess.run(sq, capture_output=True, text=True).stdout.strip()):
            die(f"job {jid} of {rd} is still {st}: it resumes by itself on requeue")
    if not (code := find_snapshot(rd)):
        die(f"no sweep code snapshot (code/GIT_COMMIT) above {rd}")
    dev = OmegaConf.select(saved, "trainer.devices")
    gpus = a.gpus or (dev if isinstance(dev, int) else int(c.gpus_per_node))
    ov = [*overrides, f"trainer.devices={gpus}", f"hydra.run.dir={rd}/hydra"]
    compose_cfg(rd, "config", ov)
    jid = submit(c, home, f"hj_resume_{saved.output_model_name}_s{saved.seed}", rd / "slurm", gpus, code,
                 ["train", "--config-path", str(rd), "--config-name", "config", *ov], a.dry)  # fmt: skip
    if not a.dry:
        record(rd, {"cmd": "resume", "overrides": overrides, "jobs": {str(rd): jid}, "code": str(code)})


def env_of(cfg: OmegaConf) -> str:  # runs made before the train configs had an `env` key fall back
    pe = OmegaConf.select(cfg, "planning_eval.config_name", default=None) or ""
    ds = str(OmegaConf.select(cfg, "data.dataset.name", default=""))
    return cfg.get("env") or pe.split("_")[0] or ("droid" if ds.endswith(".csv") else die(f"no env for {ds}"))


def cmd_eval(a, overrides: list) -> None:
    home, c = hjepa_home(), cluster("eval", a)
    root = Path(a.target).resolve()
    runs = [root] if is_run(root) else []
    for dp, dns, fns in [] if runs else os.walk(root):
        if "config.yaml" in fns and epoch_ckpts(Path(dp)):
            runs.append(Path(dp))
            dns.clear()
        dns[:] = sorted(n for n in dns if not n.startswith(("code", "hydra", ".", "slurm", "eval_", "lightning")))
    if not runs:
        die(f"no run dirs (config.yaml + *_epoch_*_object.ckpt) under {root}")
    sq = subprocess.run(["squeue", "--me", "-h", "-t", LIVE, "-o", "%j"], capture_output=True, text=True)
    active, checked, plan, ndone = set(sq.stdout.split()), set(), [], 0
    for rd in runs:
        cfg = OmegaConf.load(rd / "config.yaml")
        env, n, seed, eps = env_of(cfg), int(cfg.num_levels), int(cfg.seed), epoch_ckpts(rd)
        ecfg, run = f"{env}_{'flat' if n == 1 else f'l{n}'}", str(cfg.output_model_name)
        run += "" if run.endswith(f"s{seed}") else f"_s{seed}"
        want = sorted(eps) if a.epochs == "all" else [max(eps)] if a.epochs == "last" else \
            [int(x) for x in a.epochs.split(",")]  # fmt: skip
        for e in want:
            out, name = rd / "eval_epoch" / f"epoch_{e}", f"ev_{run}_e{e}"
            done = done_file(ecfg, out).exists()
            why = "missing" if e not in eps else "done" if done else name in active and "active"
            if why:
                ndone += done
                print(f"SKIP {name}: {why}")
                continue
            if ecfg not in checked:  # composes (catches bad overrides) and its eval tasks exist
                t = compose_cfg(H / "config/eval", ecfg, [f"seed={seed}", f"policy={eps[e]}", *overrides])
                t = t.get("load_eval_trajs_path")
                try:
                    not t or (H / t).exists() or Path(t).exists() or die(f"{ecfg}: eval tasks {t} missing")
                except PermissionError:
                    print(f"WARNING: cannot stat eval tasks {t}")
                checked.add(ecfg)
            plan.append((rd, name, f"{ecfg}|{seed}|{eps[e]}|{out}", c.eval_time.get(env, c.eval_time.default)))
    print(f"{len(runs)} runs, {ndone} evals done, {len(plan)} to run")
    if not plan:
        return
    if not (code := find_snapshot(root, "eval_code")):  # first eval of this sweep: freeze the worktree's eval code
        code = next((d for d in [root, *root.parents] if (d / "code/GIT_COMMIT").exists()), root) / "eval_code"
        print(f"{'would snapshot' if a.dry else 'snapshotting'} the worktree to {code}")
        a.dry or snapshot(code)
    ids = {}
    for rd, name, spec, t in plan:
        c.time = a.time or t
        ids[name] = submit(c, home, name, rd / "slurm", 1, code, ["eval", spec, "--", *overrides], a.dry)
    if not a.dry:
        record(root, {"cmd": "eval", "epochs": a.epochs, "overrides": overrides, "jobs": ids, "code": str(code)})


def main() -> None:
    if sys.argv[1:2] == ["_job"]:
        return run_job(sys.argv[2], sys.argv[3], sys.argv[4:])
    common = argparse.ArgumentParser(add_help=False)
    for f in ("--partition", "--account", "--qos", "--time", "--mem"):
        common.add_argument(f)
    common.add_argument("--gpus", type=int, help="GPUs per node (train: trainer.devices)")
    common.add_argument("--dry", action="store_true", help="run every check, print the sbatch lines")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train", parents=[common])
    t.add_argument("--config-name", required=True)
    t.add_argument("--sweep", required=True)
    t.add_argument("--seeds", default="42")
    t.add_argument("--grid", action="append", default=[], help="key=v1,v2 (repeatable)")
    t.add_argument("--into", help="existing sweep dir to add jobs to")
    sp.add_parser("resume", parents=[common]).add_argument("run_dir")
    e = sp.add_parser("eval", parents=[common])
    e.add_argument("target")
    e.add_argument("--epochs", default="all", help="all | last | N,M")
    a, overrides = p.parse_known_args()
    if bad := [o for o in overrides if o.startswith("-") or "=" not in o]:
        die(f"unrecognized arguments {bad} (hydra overrides are key=value)")
    {"train": cmd_train, "resume": cmd_resume, "eval": cmd_eval}[a.cmd](a, overrides)


if __name__ == "__main__":
    main()
