# Planned-vs-GT DROID end-effector path metrics, copied verbatim from the original planning code
# (the published DROID numbers): do not change the numerics.
# Actions are [T, A] raw delta-poses: 0:3 xyz, 3:6 orientation, 6 gripper.
import numpy as np
import torch


def _np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def xyz_path(actions):
    xyz = _np(actions)[:, :3].astype(float)  # [T, 3]
    return np.concatenate(
        [np.zeros((1, 3)), np.cumsum(xyz, axis=0)], axis=0
    )  # [T+1, 3]


def discrete_frechet(P, Q):
    n, m = len(P), len(Q)
    C = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)  # [n, m]
    ca = np.full((n, m), -1.0)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                ca[i, j] = C[0, 0]
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], C[i, 0])
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], C[0, j])
            else:
                ca[i, j] = max(
                    min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), C[i, j]
                )
    return float(ca[n - 1, m - 1])


def cumulative_delta(planned, gt, lo=0, hi=None):
    planned = torch.as_tensor(planned)
    gt = torch.as_tensor(gt)
    return torch.abs(planned[lo:hi].sum(0) - gt[lo:hi].sum(0))


def frechet_skill_over_floor(planned, gt):
    # skill = (floor - model) / floor, floor = Frechet distance of the do-nothing
    # (origin) path to the GT path; nan for a GT clip with no xyz motion.
    gp = xyz_path(gt)  # [T+1, 3]
    pp = xyz_path(planned)  # [T+1, 3]
    model = discrete_frechet(pp, gp)
    floor = discrete_frechet(np.zeros_like(gp), gp)  # == max_t ‖gp[t]‖
    skill = (floor - model) / floor if floor > 1e-9 else float("nan")
    return skill, model, floor
