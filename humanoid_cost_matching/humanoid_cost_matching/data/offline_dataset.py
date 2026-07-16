import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union


@dataclass
class BatchArrays:
    t0: np.ndarray  # (B,)
    x0: np.ndarray  # (B,nx)
    initMode: np.ndarray  # (B,)
    xMeasSeq: np.ndarray  # (B,N+1,nx)
    uSeq: np.ndarray  # (B,N,nu)
    tt: np.ndarray  # (B,3)
    tx: np.ndarray  # (B,3,nx)
    tu: np.ndarray  # (B,3,nu)

    def as_tuple(self):
        return (
            self.t0,
            self.x0,
            self.initMode,
            self.xMeasSeq,
            self.uSeq,
            self.tt,
            self.tx,
            self.tu,
        )


def _as_contig(a: np.ndarray, dtype) -> np.ndarray:
    """Force contiguous + dtype, without unnecessary copy if already ok."""
    if a.dtype != dtype:
        a = a.astype(dtype, copy=False)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def _load_npz(npz_or_path: Union[str, np.lib.npyio.NpzFile]):
    if isinstance(npz_or_path, str):
        return np.load(npz_or_path, allow_pickle=True)
    return npz_or_path


def valid_k_indices(
    T: int, N: int, k_indices: Optional[Sequence[int]] = None
) -> np.ndarray:
    """
    Return valid indices k such that k+N+1 <= T.
    If k_indices is None: returns all valid k in [0, T-N-1].
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if T <= 0:
        return np.zeros((0,), dtype=np.int64)

    if k_indices is None:
        # need k+N+1 <= T  ->  k <= T-N-1
        last = T - N - 1
        if last < 0:
            return np.zeros((0,), dtype=np.int64)
        return np.arange(0, last + 1, dtype=np.int64)

    k = np.asarray(k_indices, dtype=np.int64)
    k = k[(k >= 0) & (k + N + 1 <= T)]
    return k


def build_batch_arrays(
    npz_or_path: Union[str, np.lib.npyio.NpzFile],
    k_indices: Optional[Sequence[int]] = None,
    *,
    N: int = 60,
    dt: float = 0.02,
    require_keys: bool = True,
) -> Tuple[BatchArrays, np.ndarray]:
    """
    Build batch arrays for pybind OfflineQEvaluator

    Returns:
      batch: BatchArrays with shapes:
        t0: (B,)
        x0: (B,nx)
        initMode: (B,)
        xMeasSeq: (B,N+1,nx)
        uSeq: (B,N,nu)
        tt: (B,3)
        tx: (B,3,nx)
        tu: (B,3,nu)
      kept_k: (B,) the actually used indices (after dropping invalid/out-of-bound)
    """
    data = _load_npz(npz_or_path)

    keys = [
        "time",
        "state",
        "input",
        "mode",
        "target_timeTrajectory",
        "target_stateTrajectory",
        "target_inputTrajectory",
    ]
    if require_keys:
        missing = [k for k in keys if k not in data]
        if missing:
            raise KeyError(f"Missing keys in npz: {missing}")

    t = data["time"]
    X = data["state"]
    U = data["input"]
    mode = data["mode"]

    tt_all = data["target_timeTrajectory"]
    tx_all = data["target_stateTrajectory"]
    tu_all = data["target_inputTrajectory"]

    T = int(t.shape[0])
    kept_k = valid_k_indices(T, N, k_indices)
    B = int(kept_k.shape[0])
    if B == 0:
        # return empty but correctly shaped arrays
        nx = int(X.shape[1]) if X.ndim == 2 and X.shape[0] > 0 else 0
        nu = int(U.shape[1]) if U.ndim == 2 and U.shape[0] > 0 else 0
        empty = BatchArrays(
            t0=np.zeros((0,), dtype=np.float64),
            x0=np.zeros((0, nx), dtype=np.float64),
            initMode=np.zeros((0,), dtype=np.int64),
            xMeasSeq=np.zeros((0, N + 1, nx), dtype=np.float64),
            uSeq=np.zeros((0, N, nu), dtype=np.float64),
            tt=np.zeros((0, 3), dtype=np.float64),
            tx=np.zeros((0, 3, nx), dtype=np.float64),
            tu=np.zeros((0, 3, nu), dtype=np.float64),
        )
        return empty, kept_k

    # infer dims
    nx = int(X.shape[1])
    nu = int(U.shape[1])

    # allocate
    t0 = np.empty((B,), dtype=np.float64)
    x0 = np.empty((B, nx), dtype=np.float64)
    initMode = np.empty((B,), dtype=np.int64)
    xMeasSeq = np.empty((B, N + 1, nx), dtype=np.float64)
    uSeq = np.empty((B, N, nu), dtype=np.float64)

    tt = np.empty((B, 3), dtype=np.float64)
    tx = np.empty((B, 3, nx), dtype=np.float64)
    tu = np.empty((B, 3, nu), dtype=np.float64)

    # fill
    for bi, k in enumerate(kept_k):
        t0[bi] = float(t[k])
        x0[bi, :] = X[k, :]
        initMode[bi] = int(mode[k])

        xMeasSeq[bi, :, :] = X[k : k + N + 1, :]
        uSeq[bi, :, :] = U[k : k + N, :]

        tt[bi, :] = tt_all[k, :]
        tx[bi, :, :] = tx_all[k, :, :]
        tu[bi, :, :] = tu_all[k, :, :]

    # ensure contiguous (pybind requires c_style contiguous ideally)
    t0 = _as_contig(t0, np.float64)
    x0 = _as_contig(x0, np.float64)
    initMode = _as_contig(initMode, np.int64)
    xMeasSeq = _as_contig(xMeasSeq, np.float64)
    uSeq = _as_contig(uSeq, np.float64)
    tt = _as_contig(tt, np.float64)
    tx = _as_contig(tx, np.float64)
    tu = _as_contig(tu, np.float64)

    batch = BatchArrays(
        t0=t0,
        x0=x0,
        initMode=initMode,
        xMeasSeq=xMeasSeq,
        uSeq=uSeq,
        tt=tt,
        tx=tx,
        tu=tu,
    )

    # print("############# Debug: built batch arrays #############")
    # print("t0", batch.t0.shape, batch.t0.dtype)
    # print("x0", batch.x0.shape, batch.x0.dtype)
    # print("initMode", batch.initMode.shape, batch.initMode.dtype)
    # print("xMeasSeq", batch.xMeasSeq.shape, batch.xMeasSeq.dtype)
    # print("uSeq", batch.uSeq.shape, batch.uSeq.dtype)
    # print("tt", batch.tt.shape, batch.tt.dtype)
    # print("tx", batch.tx.shape, batch.tx.dtype)
    # print("tu", batch.tu.shape, batch.tu.dtype)

    # for name, arr in [
    #     ("t0", batch.t0),
    #     ("x0", batch.x0),
    #     ("xMeasSeq", batch.xMeasSeq),
    #     ("uSeq", batch.uSeq),
    #     ("tx", batch.tx),
    # ]:
    #     print(name, "contig?", arr.flags["C_CONTIGUOUS"])

    return batch, kept_k


def sample_k_indices(
    T: int,
    *,
    N: int,
    batch_size: int,
    seed: Optional[int] = None,
    replace: bool = False,
) -> np.ndarray:
    """
    Randomly sample valid k indices such that k+N <= T.
    """
    k_all = valid_k_indices(T, N, None)
    if k_all.size == 0:
        return k_all
    rng = np.random.default_rng(seed)
    if (not replace) and batch_size > k_all.size:
        batch_size = int(k_all.size)
    return rng.choice(k_all, size=batch_size, replace=replace).astype(np.int64)


def load_measured_stage_cost(
    npz_or_path: Union[str, np.lib.npyio.NpzFile],
    *,
    require_key: bool = True,
) -> np.ndarray:
    """
    Extract the 'measured_stage_cost' sequence as a contiguous float64 array.
    """
    data = _load_npz(npz_or_path)

    if "measured_stage_cost" not in data:
        if require_key:
            raise KeyError("Missing required key: 'measured_stage_cost'.")
        return np.zeros((0,), dtype=np.float64)

    l = np.asarray(data["measured_stage_cost"], dtype=np.float64)

    if l.ndim != 1:
        raise ValueError(f"Measured stage cost must be 1D, but got shape {l.shape}.")
    return _as_contig(l, np.float64)
