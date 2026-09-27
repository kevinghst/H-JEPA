"""Diff two eval-task .pt files element by element. usage: python cmp_tasks.py new.pt orig.pt"""
import sys

import numpy as np
import torch


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'new' if k in a else 'orig'}")
            else:
                walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    else:
        try:
            x = a.numpy() if torch.is_tensor(a) else np.asarray(a)
            y = b.numpy() if torch.is_tensor(b) else np.asarray(b)
            if x.shape != y.shape:
                out.append(f"{path}: shape {x.shape} vs {y.shape}")
            elif not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                out.append(f"{path}: values differ")
        except Exception:
            if a != b:
                out.append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")


out = []
walk(torch.load(sys.argv[1], weights_only=False), torch.load(sys.argv[2], weights_only=False), "", out)
print(len(out), "differences")
print("\n".join(out[:40]))
