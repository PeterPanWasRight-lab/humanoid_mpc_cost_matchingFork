import json
import os
from pathlib import Path


def _write_indexed_block(f, name, values):
    f.write(f"  {name}\n  {{\n")
    for i, v in enumerate(values):
        f.write(f"    [{i}] {float(v):.17g}\n")
    f.write("  }\n\n")


def _theta_to_blocks(theta):
    # fixed dims in your project
    nx, nu = 35, 35
    baseDim, comDim = 12, 2
    swingDim, numSwingFeet = 12, 2
    torqueDim, numTorquelegs = 6, 2

    expected = (
        6
        + nx
        + nu
        + nx
        + baseDim
        + comDim
        + numSwingFeet * swingDim
        + numTorquelegs * torqueDim
    )
    if len(theta) != expected:
        raise RuntimeError(
            f"[export_theta_info] theta dim mismatch: got {len(theta)} expected {expected}"
        )

    off = 0
    blocks = {}
    blocks["theta_hl"] = theta[off : off + 3]
    off += 3
    blocks["theta_ha"] = theta[off : off + 3]
    off += 3

    blocks["theta_q"] = theta[off : off + nx]
    off += nx
    blocks["theta_r"] = theta[off : off + nu]
    off += nu
    blocks["theta_qf"] = theta[off : off + nx]
    off += nx

    blocks["theta_base"] = theta[off : off + baseDim]
    off += baseDim
    blocks["theta_com"] = theta[off : off + comDim]
    off += comDim

    blocks["theta_swing_foot0"] = theta[off : off + swingDim]
    off += swingDim
    blocks["theta_swing_foot1"] = theta[off : off + swingDim]
    off += swingDim

    blocks["theta_torque_leg0"] = theta[off : off + torqueDim]
    off += torqueDim
    blocks["theta_torque_leg1"] = theta[off : off + torqueDim]
    off += torqueDim

    return blocks


def export_theta_to_info(
    theta_json_path: str,
    run_out_dir: str,
    deployed_dir: str,
    run_id: str,
    step: int,
    enable: bool = True,
):
    theta_json_path = Path(theta_json_path)
    run_out_dir = Path(run_out_dir)
    deployed_dir = Path(deployed_dir)
    deployed_hist = deployed_dir / "history"

    run_out_dir.mkdir(parents=True, exist_ok=True)
    deployed_dir.mkdir(parents=True, exist_ok=True)
    deployed_hist.mkdir(parents=True, exist_ok=True)

    data = json.loads(theta_json_path.read_text())
    theta = data.get("theta", None)
    if theta is None:
        raise RuntimeError(
            f"[export_theta_info] missing key 'theta' in {theta_json_path}"
        )

    blocks = _theta_to_blocks(theta)

    # -------- run-local info (versioned) --------
    run_info = run_out_dir / f"theta_step_{step:08d}.info"
    tmp = run_info.with_suffix(".info.tmp")
    with tmp.open("w") as f:
        f.write("cost_matching\n{\n")
        f.write(f"  enable {'true' if enable else 'false'}\n\n")

        _write_indexed_block(f, "theta_hl", blocks["theta_hl"])
        _write_indexed_block(f, "theta_ha", blocks["theta_ha"])

        f.write("  tracking\n  {\n")
        _write_indexed_block(f, "theta_q", blocks["theta_q"])
        _write_indexed_block(f, "theta_r", blocks["theta_r"])
        _write_indexed_block(f, "theta_qf", blocks["theta_qf"])
        f.write("  }\n\n")

        _write_indexed_block(f, "theta_base", blocks["theta_base"])
        _write_indexed_block(f, "theta_com", blocks["theta_com"])

        f.write("  theta_swing\n  {\n")
        _write_indexed_block(f, "foot0", blocks["theta_swing_foot0"])
        _write_indexed_block(f, "foot1", blocks["theta_swing_foot1"])
        f.write("  }\n\n")

        f.write("  theta_torque\n  {\n")
        _write_indexed_block(f, "leg0", blocks["theta_torque_leg0"])
        _write_indexed_block(f, "leg1", blocks["theta_torque_leg1"])
        f.write("  }\n\n")

        f.write("}\n")
    os.replace(tmp, run_info)

    # -------- deployed history (versioned by run_id + step) --------
    deployed_info_hist = deployed_hist / f"{run_id}_step_{step:08d}.info"
    tmp_h = deployed_info_hist.with_suffix(".info.tmp")
    tmp_h.write_text(run_info.read_text())
    os.replace(tmp_h, deployed_info_hist)

    # -------- atomically update deployed latest --------
    deployed_latest = deployed_dir / "learned_theta.info"
    tmp_latest = deployed_dir / "learned_theta.info.tmp"
    tmp_latest.write_text(run_info.read_text())
    os.replace(tmp_latest, deployed_latest)
