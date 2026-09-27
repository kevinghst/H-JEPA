"""Diff two planning-eval metrics.yaml files (timing keys ignored). usage: python cmp_metrics.py a.yaml b.yaml"""
import math
import sys

import yaml

a, b = yaml.safe_load(open(sys.argv[1])), yaml.safe_load(open(sys.argv[2]))
for k in sorted(set(a) | set(b)):
    if "evaluation_time" in k or "wall_clock" in k or k == "peak_gpu_mem_bytes":
        continue
    x, y = a.get(k, "<none>"), b.get(k, "<none>")
    same = x == y or (isinstance(x, float) and isinstance(y, float)
                      and ((math.isnan(x) and math.isnan(y)) or abs(x - y) <= 1e-6 * max(1, abs(x))))
    print(("  same " if same else "  DIFF ") + f"{k}: {str(x)[:60]} | {str(y)[:60]}")
