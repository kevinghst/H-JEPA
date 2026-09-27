"""Diff two eval configs, ignoring run-specific keys. usage: python cdiff.py a.yaml b.yaml"""
import sys

import yaml


def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, f"{p}.{k}" if p else str(k)))
    else:
        o[p] = d
    return o


IGN = ("hydra", "output", "sweep", "wandb", "policy", "seed", "save_video", "save_plots", "cache_dir", "n_gpus", "cpus")


def diff(a, b, ign=IGN):
    A, B = flat(yaml.safe_load(open(a))), flat(yaml.safe_load(open(b)))
    out = []
    for k in sorted(set(A) | set(B)):
        if any(k.startswith(i) or ("." + i) in k for i in ign):
            continue
        if A.get(k, "<none>") != B.get(k, "<none>"):
            out.append(f"  {k}: {A.get(k, '<none>')} | {B.get(k, '<none>')}")
    return out


if __name__ == "__main__":
    print("\n".join(diff(sys.argv[1], sys.argv[2])) or "  IDENTICAL")
