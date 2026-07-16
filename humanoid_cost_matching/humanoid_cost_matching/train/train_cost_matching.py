import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np

from humanoid_cost_matching.data.offline_dataset import (
    build_batch_arrays,
    load_measured_stage_cost,
    valid_k_indices,
)
from humanoid_cost_matching.utils.export_theta_info import export_theta_to_info

import humanoid_cost_matching_offline_eval_py as offeval


# ----------------------------- constants (fixed by task.info) -----------------------------
DYN_DIM = 6  # theta_hl(3), theta_ha(3)
BASE_DIM = 12
COM_DIM = 2
SWING_DIM = 2 * 12
TORQUE_DIM = 2 * 6


# ----------------------------- utils: theta layout dim -----------------------------


def compute_theta_dim(nx: int, nu: int) -> int:
    """
    Must match C++ ThetaLayout dim_total.
    NOTE:If later change C++ layout, update here accordingly.
    """
    # layout: dyn(6) + Q(nx) + R(nu) + Qf(nx) + base(12) + com(2) + swing(24) + torque(12)
    return DYN_DIM + nx + nu + nx + BASE_DIM + COM_DIM + SWING_DIM + TORQUE_DIM


# ----------------------------- utils: softplus parameterization -----------------------------


def softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    # softplus(x)=log(1+exp(x))
    out = np.empty_like(x, dtype=np.float64)
    # for large x: softplus ~ x; for small: exp(x)
    th = 20.0
    mask = x > th
    out[mask] = x[mask]
    out[~mask] = np.log1p(np.exp(x[~mask]))
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def raw_to_theta(raw: np.ndarray, eps: float) -> np.ndarray:
    return softplus(raw) + eps


def dtheta_draw(raw: np.ndarray) -> np.ndarray:
    # derivative of softplus is sigmoid
    return sigmoid(raw)


# ----------------------------- utils: returns -----------------------------


def discounted_return_to_go(stage_cost: np.ndarray, gamma: float) -> np.ndarray:
    """
    Q[k] = sum_{i=0}^{T-k-1} gamma^i * stage_cost[k+i]

    stage_cost here should already be in the same "per-step" unit

    TODO(dt):
      Decide whether stage_cost should be multiplied by dt.
      - If stage_cost is cost-rate: use stage_cost_dt = stage_cost * dt before calling here.
      - If already discrete: keep as is.
    """
    stage_cost = np.asarray(stage_cost, dtype=np.float64)
    T = stage_cost.shape[0]
    Q = np.zeros((T + 1,), dtype=np.float64)
    for k in range(T - 1, -1, -1):
        Q[k] = stage_cost[k] + gamma * Q[k + 1]
    return Q[:-1]


# ----------------------------- optimizer: Adam -----------------------------


class Adam:
    def __init__(
        self,
        dim: int,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.m = np.zeros((dim,), dtype=np.float64)
        self.v = np.zeros((dim,), dtype=np.float64)

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * g
        self.v = self.b2 * self.v + (1.0 - self.b2) * (g * g)
        mhat = self.m / (1.0 - (self.b1**self.t))
        vhat = self.v / (1.0 - (self.b2**self.t))
        x = x - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return x


# ----------------------------- io -----------------------------


def save_checkpoint(path: str, raw_theta: np.ndarray, step: int, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": int(step),
        "raw_theta": raw_theta.tolist(),
        "meta": meta,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_checkpoint(path: str) -> Tuple[int, np.ndarray, dict]:
    with open(path, "r") as f:
        d = json.load(f)
    step = int(d["step"])
    raw_theta = np.asarray(d["raw_theta"], dtype=np.float64)
    meta = d.get("meta", {})
    return step, raw_theta, meta


def save_theta(path: str, theta: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"theta": theta.tolist(), "meta": meta}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ----------------------------- dataset wrapper -----------------------------


@dataclass
class TrajData:
    path: str
    T: int
    Qmeas_all: np.ndarray  # shape (T,)


def load_trajectory(
    npz_path: str, gamma: float, dt: float, cost_is_rate: bool
) -> TrajData:
    l = load_measured_stage_cost(npz_path, require_key=True)  # (T,)
    # TODO(dt): confirm whether to multiply dt here
    if cost_is_rate:
        l_used = l * dt
    else:
        l_used = l
    Q = discounted_return_to_go(l_used, gamma)
    return TrajData(path=npz_path, T=int(l.shape[0]), Qmeas_all=Q)


def list_npz_files(npz_arg: List[str]) -> List[str]:
    """
    Accept:
      - multiple --npz file paths
      - directories (will scan *.npz)
    """
    out: List[str] = []
    for item in npz_arg:
        if os.path.isdir(item):
            for fn in sorted(os.listdir(item)):
                if fn.endswith(".npz"):
                    out.append(os.path.join(item, fn))
        else:
            out.append(item)
    # unique keep order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


# ------------- default paths -------------
def _make_run_id():
    return time.strftime("run_%Y%m%d_%H%M%S", time.localtime())


def _default_run_dir(run_id: str):
    return (
        "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_cost_matching/logs/cm_runs/"
        + run_id
    )


def _deployed_theta_dir():
    return "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_cost_matching/logs/deployed_theta"


# ----------------------------- training core -----------------------------


def main():
    ap = argparse.ArgumentParser()

    # evaluator config
    ap.add_argument(
        "--task",
        default="/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_cost_matching.info",
    )
    ap.add_argument(
        "--urdf",
        default="/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
    )
    ap.add_argument(
        "--ref",
        default="/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.info",
    )

    # dataset
    ap.add_argument(
        "--npz",
        nargs="+",
        help="npz files and/or directories",
        default=[
            "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_cost_matching/logs/obs_data"
        ],
    )
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--N", type=int, default=60)

    # dt handling
    ap.add_argument(
        "--cost_is_rate",
        action="store_true",
        default=True,
        help="If set, measured_stage_cost is treated as cost-rate and multiplied by dt. TODO confirm.",
    )
    ap.add_argument("--no_cost_is_rate", dest="cost_is_rate", action="store_false")

    # training
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--grad_clip",
        type=float,
        default=10.0,
        help="clip L2 norm of grad_raw (0 disables)",
    )
    ap.add_argument("--theta_eps", type=float, default=1e-8)
    ap.add_argument(
        "--raw_init",
        type=float,
        default=0.541324854,
        help="raw theta init; theta=softplus(raw)+eps",
    )

    # logging / ckpt
    ap.add_argument(
        "--outdir",
        help="output directory for logs and checkpoints",
    )
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--resume", type=str, default="", help="path to checkpoint json")
    ap.add_argument("--tag", type=str, default="", help="optional tag for runs")

    args = ap.parse_args()

    # ----- ensure output dirs -----
    run_id = _make_run_id()
    if not getattr(args, "outdir", None):
        args.outdir = _default_run_dir(run_id)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(_deployed_theta_dir()).mkdir(parents=True, exist_ok=True)
    Path(_deployed_theta_dir() + "/history").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ---- load trajectories ----
    npz_files = list_npz_files(args.npz)
    if not npz_files:
        raise RuntimeError("No npz found.")

    trajs: List[TrajData] = []
    for p in npz_files:
        td = load_trajectory(
            p, gamma=args.gamma, dt=args.dt, cost_is_rate=args.cost_is_rate
        )
        trajs.append(td)

    # ---- build evaluator (C++) ----
    ev = offeval.OfflineQEvaluator(args.task, args.urdf, args.ref)

    # ---- infer theta dim from nx,nu in the first traj (consistent dims across trajs) ----
    first = np.load(trajs[0].path, allow_pickle=True)
    nx = int(first["state"].shape[1])
    nu = int(first["input"].shape[1])
    p = compute_theta_dim(nx, nu)

    # ---- init / resume raw_theta ----
    step0 = 0
    raw_theta = np.full((p,), float(args.raw_init), dtype=np.float64)

    if args.resume:
        step0, raw_theta_loaded, meta = load_checkpoint(args.resume)
        raw_theta = raw_theta_loaded.astype(np.float64, copy=False)
        if raw_theta.shape[0] != p:
            raise RuntimeError(
                f"Resume theta dim mismatch: got {raw_theta.shape[0]} expected {p}"
            )

    opt = Adam(p, lr=args.lr)

    # ---- logging ----
    log_path = os.path.join(
        args.outdir, f"train_log{('_'+args.tag) if args.tag else ''}.csv"
    )
    if step0 == 0 and (not os.path.exists(log_path)):
        with open(log_path, "w") as f:
            f.write("step,loss,grad_norm,theta_min,theta_mean,theta_max,wall_s\n")

    t_wall0 = time.time()

    # Precompute per-traj valid ks (k+N <= T) for sampling
    valid_ks_per_traj = []
    for td in trajs:
        valid_ks_per_traj.append(valid_k_indices(td.T, args.N, None))

    # ---- training loop ----
    for step in range(step0 + 1, args.iters + 1):
        # sample traj ids
        # (uniform over trajectories; TODO: weight by length if you want)
        traj_ids = rng.integers(0, len(trajs), size=args.batch)

        # For each sampled traj, sample a valid k inside it
        ks = np.empty((args.batch,), dtype=np.int64)
        for i, tid in enumerate(traj_ids):
            k_all = valid_ks_per_traj[tid]
            if k_all.size == 0:
                ks[i] = 0
                continue
            ks[i] = int(k_all[rng.integers(0, k_all.size)])

        # Build batch per-trajectory group to avoid Npz reopen B times (speed)
        # Strategy:
        #   group indices by traj, call build_batch_arrays per group, then concat
        unique_tids = np.unique(traj_ids)
        batches = []
        kept_ks_all = []
        Qmeas_targets = []

        for tid in unique_tids:
            mask = np.where(traj_ids == tid)[0]
            k_list = ks[mask].tolist()
            batch, kept_k = build_batch_arrays(
                trajs[tid].path, k_list, N=args.N, dt=args.dt, require_keys=True
            )
            batches.append((mask, batch, kept_k))
            # measured targets
            Qmeas_targets.append(trajs[tid].Qmeas_all[kept_k])

        # concat batches into one BatchArrays (stable ordering by mask indices)
        # allocate merged arrays
        B = args.batch
        t0 = np.empty((B,), dtype=np.float64)
        x0 = np.empty((B, nx), dtype=np.float64)
        initMode = np.empty((B,), dtype=np.int64)
        xMeasSeq = np.empty((B, args.N + 1, nx), dtype=np.float64)
        uSeq = np.empty((B, args.N, nu), dtype=np.float64)
        tt = np.empty((B, 3), dtype=np.float64)
        tx = np.empty((B, 3, nx), dtype=np.float64)
        tu = np.empty((B, 3, nu), dtype=np.float64)

        Qmeas = np.empty((B,), dtype=np.float64)

        for (mask, batch, kept_k), qtar in zip(batches, Qmeas_targets):
            t0[mask] = batch.t0
            x0[mask, :] = batch.x0
            initMode[mask] = batch.initMode
            xMeasSeq[mask, :, :] = batch.xMeasSeq
            uSeq[mask, :, :] = batch.uSeq
            tt[mask, :] = batch.tt
            tx[mask, :, :] = batch.tx
            tu[mask, :, :] = batch.tu
            Qmeas[mask] = qtar

        # -------- theta positive (softplus) --------
        theta = raw_to_theta(raw_theta, eps=args.theta_eps)

        # -------- evaluate Q and dQ/dtheta from C++ --------
        Qmpc, dQ_dtheta = ev.evaluate_q_and_grad_batch(
            t0, x0, initMode, xMeasSeq, uSeq, tt, tx, tu, float(args.dt), theta
        )

        # -------- loss --------
        diff = Qmpc - Qmeas
        loss = float(np.mean(diff * diff))

        # dL/dtheta = mean( 2*diff * dQ/dtheta )
        dL_dtheta = np.mean((2.0 * diff)[:, None] * dQ_dtheta, axis=0)

        # chain rule to raw: dtheta/draw = sigmoid(raw)
        dtheta = dtheta_draw(raw_theta)
        grad_raw = dL_dtheta * dtheta

        # -------- grad clip --------
        if args.grad_clip and args.grad_clip > 0.0:
            gn = float(np.linalg.norm(grad_raw))
            if gn > args.grad_clip:
                grad_raw = grad_raw * (args.grad_clip / (gn + 1e-12))
        grad_norm = float(np.linalg.norm(grad_raw))

        # -------- update raw_theta --------
        raw_theta = opt.step(raw_theta, grad_raw)

        # -------- logging --------
        if (step % args.log_every == 0) or (step == 1):
            wall = time.time() - t_wall0
            theta_now = raw_to_theta(raw_theta, eps=args.theta_eps)
            row = (
                f"{step},{loss:.8e},{grad_norm:.6e},"
                f"{theta_now.min():.6e},{theta_now.mean():.6e},{theta_now.max():.6e},"
                f"{wall:.2f}\n"
            )
            with open(log_path, "a") as f:
                f.write(row)
            print(
                f"[{step:06d}] loss={loss:.6e} |grad_raw|={grad_norm:.3e} "
                f"theta(min/mean/max)=({theta_now.min():.2e}/{theta_now.mean():.2e}/{theta_now.max():.2e})"
            )
            print(f"  sample Qmpc: {Qmpc[:2]} | Qmeas: {Qmeas[:2]} | diff: {diff[:2]}")

        # -------- checkpoint --------
        if (step % args.save_every == 0) or (step == args.iters):
            meta = {
                "task": args.task,
                "urdf": args.urdf,
                "ref": args.ref,
                "npz": npz_files,
                "gamma": args.gamma,
                "dt": args.dt,
                "N": args.N,
                "cost_is_rate": args.cost_is_rate,
                "theta_eps": args.theta_eps,
                "theta_dim": p,
            }

            # ckpt (raw theta)
            ckpt_path = os.path.join(args.outdir, f"ckpt_step_{step:08d}.json")
            save_checkpoint(ckpt_path, raw_theta=raw_theta, step=step, meta=meta)

            # theta_latest.json (POSITIVE theta, already softplus)
            theta_path = os.path.join(args.outdir, "theta_latest.json")
            save_theta(
                theta_path, theta=raw_to_theta(raw_theta, eps=args.theta_eps), meta=meta
            )

            # -------- export to .info and deploy to stable path for MPC --------
            export_theta_to_info(
                theta_json_path=theta_path,
                run_out_dir=args.outdir,
                deployed_dir=_deployed_theta_dir(),
                run_id=run_id,
                step=step,
                enable=True,
            )

    # final
    meta = {
        "task": args.task,
        "urdf": args.urdf,
        "ref": args.ref,
        "npz": npz_files,
        "gamma": args.gamma,
        "dt": args.dt,
        "N": args.N,
        "cost_is_rate": args.cost_is_rate,  # TODO confirm
        "theta_eps": args.theta_eps,
        "theta_dim": p,
    }
    save_theta(
        os.path.join(args.outdir, "theta_final.json"),
        raw_to_theta(raw_theta, eps=args.theta_eps),
        meta=meta,
    )
    print("Saved theta_final.json")


if __name__ == "__main__":
    main()
