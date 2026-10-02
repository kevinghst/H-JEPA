"""SLURM launcher for H-JEPA training, resume and per-epoch planning eval. Run from h_jepa/.

  python launch.py train --config-name cube_lewm --sweep crop_ab [--seeds 42,43,44] \\
      [--grid level1.wm.history_size=3,7 ...] [--into <sweep_dir>] [hydra overrides ...] [--dry]
  python launch.py resume <run_dir> [hydra overrides, e.g. trainer.max_epochs=100] [--dry]
  python launch.py eval <sweep_dir|run_dir> [--epochs all|last|N,M] [--chunk K] [--tasks <abs .pt>] \\
      [eval.py overrides, e.g. eval.num_eval=4] [--dry]
  common flags: --partition P --account A --qos Q --time HH:MM:SS --gpus N --mem 200G (partition and account go together)

train   One job per (grid cell x seed): the cartesian product of every --grid key=v1,v2 and --seeds.
        Sweep dir $HJEPA_HOME/ckpts/<env>/<sweep>_<YYYY-MM-DD_HH-MM>/ (<env> = the config's env key),
        run dirs <sweep_dir>/<cell>/seed<S>,
        output_model_name = <cell>, a readable token per grid key: level2.wm.history_size=7 -> l2hs7
        (level number + initials of the last key part; full key when two tokens would collide).
        The tree (git ls-files -co --exclude-standard) is copied to <sweep_dir>/code with GIT_COMMIT,
        GIT_STATUS and UNCOMMITTED.diff; jobs run from that copy, so later worktree edits never change
        the sweep. --into <sweep_dir> adds cells/seeds to an existing sweep with its own code copy.
        wandb (when the cfg enables it) gets group = sweep name. <sweep_dir>/launch.json logs each launch.
resume  Relaunch one run from its saved <run_dir>/config.yaml (main_hjepa.py --config-path <run_dir>
        --config-name config), from the sweep's code copy when there is one. Training restarts from
        lightning_resume/last.ckpt (full state). Refused for a sweep dir, a run without config.yaml or
        last.ckpt (main_hjepa.py cannot resume from epoch ckpts alone), or a run with a live job.
eval    Planning eval of every <run>/<name>_epoch_<N>_object.ckpt under the target with
        eval.py --config-name <env>_{flat|l2|l3|l4} (planner = the run's num_levels, env from the saved
        planning_eval.config_name or the dataset name), planner seed = model seed, output
        <run>/eval_epoch/epoch_<N>/metrics.yaml. Done evals and live jobs are skipped, so re-running
        picks up new epochs; --chunk K epochs per job; --time default = base + per-epoch minutes x K.

Before any sbatch: HJEPA_HOME set; the Hydra config composes with the exact overrides (unknown keys
fail); dataset files named in cfg.data exist; fresh train run dirs are absent; cluster settings are
complete. Cluster settings: config/slurm/default.yaml, overridden by config/slurm/local.yaml (gitignored:
venv, partition, account, ...), overridden by CLI flags. Job logs: <run_dir>/slurm/%x_%j.out.
Job body: scripts/launch.sbatch (--requeue, --open-mode=append; multi-GPU via srun, one task per GPU).
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
SBATCH = H / "scripts" / "launch.sbatch"
RESERVED = {"", "sweep", "test", "tmp", "default"}
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
    c = OmegaConf.merge(c, c.get(sub) or {})
    if a.partition and not a.account:
        die("--partition needs --account too (they are set together)")
    for k in ("partition", "account", "qos", "time", "mem"):
        if getattr(a, k):
            c[k] = getattr(a, k)
    if a.gpus:
        c.gpus_per_node = a.gpus
    keys = ("venv", "partition", "account", "qos", "gpus_per_node", "cpus_per_task", "mem", "time")
    if missing := [k for k in keys if c.get(k) in (None, "")]:
        die(f"cluster settings missing {missing}: set them in {H}/config/slurm/local.yaml")
    if not Path(c.venv, "bin/activate").exists():
        die(f"venv {c.venv} has no bin/activate")
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
        v = OmegaConf.select(cfg, f"data.dataset.{k}", default=None)
        if not v:
            continue
        p = home / "droid" / v if v.endswith(".csv") else home / f"{v}.h5"
        if not p.exists():
            die(f"data.dataset.{k}={v}: {p} not found")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def snapshot(dest: Path) -> None:
    dest.mkdir(parents=True)
    files = git("ls-files", "-co", "--exclude-standard")
    rsync = ["rsync", "-a", "--ignore-missing-args", "--files-from=-", ".", str(dest)]
    subprocess.run(rsync, cwd=REPO, input=files, text=True, check=True)
    if os.path.lexists(H / "assets") and not os.path.lexists(dest / "h_jepa/assets"):
        (dest / "h_jepa/assets").symlink_to(os.path.realpath(H / "assets"))  # gitignored eval tasks
    (dest / "GIT_COMMIT").write_text(git("rev-parse", "HEAD"))
    (dest / "GIT_STATUS").write_text(git("status", "--short"))
    (dest / "UNCOMMITTED.diff").write_text(git("diff", "HEAD"))
    print(f"code snapshot {dest} @ {git('rev-parse', '--short', 'HEAD').strip()}")


def find_snapshot(p: Path, home: Path) -> Path | None:
    for d in [p, *p.parents]:
        if (d / "code/GIT_COMMIT").exists():
            code = d / "code"
            if (code / "GIT_COMMIT").read_text() != git("rev-parse", "HEAD") or (
                code / "UNCOMMITTED.diff"
            ).read_text() != git("diff", "HEAD"):
                print(f"WARNING: {code} differs from the worktree; jobs run the snapshot, not your edits")
            return code
        if d == home / "ckpts":
            return None
    return None


def code_root(p: Path, home: Path, dry: bool, tag: str) -> Path:
    """The sweep's code snapshot above p, else a fresh snapshot at p/code_<tag>_<ts>."""
    if code := find_snapshot(p, home):
        return code
    code = p / f"code_{tag}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    print(f"no sweep code snapshot above {p}: {'would snapshot' if dry else 'snapshotting'} the worktree")
    if not dry:
        snapshot(code)
    return code


def epoch_ckpts(d: Path) -> dict:
    return {int(m.group(1)): c for c in d.glob("*_epoch_*_object.ckpt") if (m := EPOCH_RE.search(c.name))}


def is_run(d: Path) -> bool:
    return (d / "config.yaml").exists() and bool(epoch_ckpts(d) or (d / "lightning_resume").exists())


def active_job_names() -> set:
    r = subprocess.run(["squeue", "--me", "-h", "-t", LIVE, "-o", "%j"], capture_output=True, text=True)
    return set(r.stdout.split())


def submit(c, home: Path, name: str, log_dir: Path, gpus: int, code: Path, args: list, dry: bool) -> str | None:
    cmd = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={c.partition}",
           f"--account={c.account}", f"--qos={c.qos}", f"--time={c.time}", f"--ntasks-per-node={gpus}",
           f"--gpus-per-node={gpus}", f"--cpus-per-task={c.cpus_per_task}", f"--mem={c.mem}",
           f"--output={log_dir}/%x_%j.out",
           f"--export=ALL,HJ_CODE={code},HJ_VENV={c.venv},HJ_GPUS={gpus},HJEPA_HOME={home}",
           str(code / "h_jepa/scripts/launch.sbatch" if not dry else SBATCH), *args]  # fmt: skip
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
    if a.sweep.lower() in RESERVED or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.sweep):
        die(f"--sweep {a.sweep!r}: give a descriptive, filesystem-safe name (not {sorted(RESERVED - {''})})")
    for o in overrides:
        if (k := o.split("=")[0].lstrip("+~")) in SET_BY_LAUNCHER or k.startswith("hydra."):
            die(f"{k} is set by launch.py")
    grid = []
    for g in a.grid:
        k, _, vs = g.partition("=")
        if not vs or k.lstrip("+~") in SET_BY_LAUNCHER:
            die(f"--grid {g!r}: expected key=v1,v2 on a key launch.py does not set")
        grid.append((k, vs.split(",")))
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
        find_snapshot(sweep_dir, home)
    elif not a.dry:
        snapshot(code)
    ids = {}
    for cell, s, run_dir, ov in jobs:
        name = f"hj_{a.sweep}_{cell}_s{s}"
        args = ["train", "--config-name", a.config_name, *ov]
        ids[str(run_dir)] = submit(c, home, name, run_dir / "slurm", c.gpus_per_node, code, args, a.dry)
    if not a.dry:
        commit = (code / "GIT_COMMIT").read_text().strip()
        record(sweep_dir, {"cmd": "train", "config_name": a.config_name, "overrides": overrides,
                           "grid": dict(grid), "seeds": a.seeds, "jobs": ids, "snapshot_commit": commit})


def cmd_resume(a, overrides: list) -> None:
    home, c = hjepa_home(), cluster("resume", a)
    rd = Path(a.run_dir).resolve()
    below = [d for d in [*rd.glob("*"), *rd.glob("*/*")] if d.is_dir() and is_run(d)]
    if below:
        die(f"{rd} contains {len(below)} run dirs (a sweep dir?): resume one run dir")
    if not (rd / "config.yaml").exists():
        die(f"{rd} has no config.yaml: not a run dir (not started yet?)")
    if not (rd / "lightning_resume/last.ckpt").exists():
        extra = " (epoch ckpts exist, but main_hjepa.py resumes only from last.ckpt)" if epoch_ckpts(rd) else ""
        die(f"nothing to resume in {rd}: no lightning_resume/last.ckpt{extra}; start a new run with train")
    saved = OmegaConf.load(rd / "config.yaml")
    if (home / "ckpts" / str(OmegaConf.select(saved, "subdir"))).resolve() != rd:
        die(f"saved subdir={OmegaConf.select(saved, 'subdir')} is not {rd} under HJEPA_HOME={home}")
    for f in (rd / "launch.json", rd.parent.parent / "launch.json"):
        log = json.loads(f.read_text()) if f.exists() else []
        if not (jid := next((e["jobs"][str(rd)] for e in reversed(log) if str(rd) in e.get("jobs", {})), None)):
            continue
        r = subprocess.run(["squeue", "-h", "-j", str(jid), "-t", LIVE, "-o", "%T"], capture_output=True, text=True)
        if r.stdout.strip():
            die(f"job {jid} of {rd} is still {r.stdout.strip()}: it resumes by itself on requeue")
    name = f"hj_resume_{saved.output_model_name}_s{saved.seed}"
    if live := {name, str(saved.output_model_name)} & active_job_names():
        die(f"a job named {live} is queued or running for {rd}")
    dev = OmegaConf.select(saved, "trainer.devices")
    gpus = a.gpus or (dev if isinstance(dev, int) else int(c.gpus_per_node))
    ov = [*overrides, f"trainer.devices={gpus}", f"hydra.run.dir={rd}/hydra"]
    compose_cfg(rd, "config", ov)
    code = code_root(rd, home, a.dry, "resume")
    jid = submit(c, home, name, rd / "slurm", gpus, code, ["train", "--config-path", str(rd),
                 "--config-name", "config", *ov], a.dry)  # fmt: skip
    if not a.dry:
        record(rd, {"cmd": "resume", "overrides": overrides, "jobs": {str(rd): jid}, "code": str(code)})


def env_of(cfg: OmegaConf) -> str:
    envs = {p.name.split("_")[0] for p in (H / "config/eval").glob("*.yaml")}
    ds = Path(str(OmegaConf.select(cfg, "data.dataset.name", default=""))).name.lower()
    cands = [str(OmegaConf.select(cfg, "planning_eval.config_name", default="") or "").split("_")[0],
             *re.split(r"[^a-z]+", ds)]  # fmt: skip
    if env := next((e for e in cands if e in envs), None):
        return env
    die(f"cannot tell the env of the run (candidates {cands}, eval envs {sorted(envs)})")


def cmd_eval(a, overrides: list) -> None:
    home, c = hjepa_home(), cluster("eval", a)
    root = Path(a.target).resolve()
    if a.tasks and not (Path(a.tasks).is_absolute() and Path(a.tasks).is_file()):
        die(f"--tasks {a.tasks}: expected an existing absolute .pt path")
    runs = [root] if is_run(root) else []
    for dp, dns, fns in [] if runs else os.walk(root):
        if "config.yaml" in fns and epoch_ckpts(Path(dp)):
            runs.append(Path(dp))
            dns.clear()
        dns[:] = sorted(n for n in dns if not n.startswith(("code", "hydra", ".", "slurm", "eval_", "lightning")))
    if not runs:
        die(f"no run dirs (config.yaml + *_epoch_*_object.ckpt) under {root}")
    active, checked, plan, ndone = active_job_names(), set(), [], 0
    for rd in runs:
        cfg = OmegaConf.load(rd / "config.yaml")
        env, n, seed = env_of(cfg), int(cfg.num_levels), int(cfg.seed)
        ecfg = f"{env}_{'flat' if n == 1 else f'l{n}'}"
        eps = epoch_ckpts(rd)
        want = sorted(eps) if a.epochs == "all" else [max(eps)] if a.epochs == "last" else \
            [int(x) for x in a.epochs.split(",")]  # fmt: skip
        todo = []
        for e in want:
            out = rd / "eval_epoch" / f"epoch_{e}"
            if e not in eps:
                print(f"MISSING {rd} epoch {e}")
            elif (out / "metrics.yaml").exists():
                ndone += 1
            else:
                todo.append((e, f"{ecfg}|{seed}|{eps[e]}|{out}|{a.tasks or ''}"))
        if todo and ecfg not in checked:
            tasks = [f"load_eval_trajs_path={a.tasks}"] if a.tasks else []
            ecomp = compose_cfg(H / "config/eval", ecfg, [f"seed={seed}", f"policy={eps[todo[0][0]]}",
                                f"output.dir={rd}", *tasks, *overrides])  # fmt: skip
            if not a.tasks and (t := ecomp.get("load_eval_trajs_path")):
                try:
                    ok = (H / t).exists()
                except PermissionError:
                    ok = print(f"WARNING: cannot stat default tasks {H / t}") or True
                if not ok:
                    die(f"{ecfg}: default tasks {H / t} missing; pass --tasks")
            checked.add(ecfg)
        for i in range(0, len(todo), a.chunk):
            chunk = todo[i : i + a.chunk]
            name = f"ev_{cfg.output_model_name}{'' if cfg.output_model_name.endswith(f's{seed}') else f'_s{seed}'}_e{'-'.join(str(e) for e, _ in chunk)}"
            if name in active:
                print(f"ACTIVE {name}: skipped")
                continue
            per = c.minutes_per_epoch.get(env, c.minutes_per_epoch.default)
            m = c.base_minutes + per * len(chunk)
            plan.append((rd, name, [s for _, s in chunk], a.time or f"{m // 60:02d}:{m % 60:02d}:00"))
    print(f"{len(runs)} runs, {ndone} evals done, {sum(len(p[2]) for p in plan)} to run in {len(plan)} jobs")
    if not plan:
        return
    code = code_root(root, home, a.dry, "eval")
    ids = {}
    for rd, name, specs, t in plan:
        c.time = t
        ids[name] = submit(c, home, name, rd / "slurm", 1, code, ["eval", *specs, "--", *overrides], a.dry)
    if not a.dry:
        record(root, {"cmd": "eval", "epochs": a.epochs, "tasks": a.tasks, "overrides": overrides,
                      "jobs": ids, "code": str(code)})  # fmt: skip


def main() -> None:
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
    r = sp.add_parser("resume", parents=[common])
    r.add_argument("run_dir")
    e = sp.add_parser("eval", parents=[common])
    e.add_argument("target")
    e.add_argument("--epochs", default="all", help="all | last | N,M")
    e.add_argument("--chunk", type=int, default=1, help="epochs per job")
    e.add_argument("--tasks", help="absolute eval tasks .pt (load_eval_trajs_path)")
    a, overrides = p.parse_known_args()
    if bad := [o for o in overrides if o.startswith("-") or "=" not in o]:
        die(f"unrecognized arguments {bad} (hydra overrides are key=value)")
    {"train": cmd_train, "resume": cmd_resume, "eval": cmd_eval}[a.cmd](a, overrides)


if __name__ == "__main__":
    main()
