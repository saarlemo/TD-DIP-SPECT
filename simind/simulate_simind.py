#!/usr/bin/env python3
"""Run one SIMIND projection simulation per XCAT phase and detector head.

Set SIMIND_PATH to the SIMIND installation directory containing smc_dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PHASES = 256
VOXELS = 256 * 256 * 306
# The reference thorax contains 210 MBq and 11,681,335 XCAT activity units.
SOURCE_SCALE_MBQ = 210.0 / 11_681_335


def ranmar_state(ij: int, kl: int) -> list[float]:
    """Create a RANMAR state from a distinct seed pair."""
    i = (ij // 177) % 177 + 2
    j = ij % 177 + 2
    k = (kl // 169) % 178 + 1
    l = kl % 169
    values = []
    for _ in range(97):
        value, bit = 0.0, 0.5
        for _ in range(24):
            m = ((i * j) % 179) * k % 179
            i, j, k = j, k, m
            l = (53 * l + 1) % 169
            if (l * m) % 64 >= 32:
                value += bit
            bit *= 0.5
        values.append(value)
    return values + [362436 / 16777216, 7654321 / 16777216,
                     16777213 / 16777216, 97, 33]


def configure_smc(template: Path, run_dir: Path, head: int, activity_mbq: float) -> None:
    lines = template.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "# Basic Change data" in line) + 1
    stop = next(i for i, line in enumerate(lines) if "# Simulation flags" in line)
    values = [float(row[i:i + 12]) for row in lines[start:stop]
              for i in range(0, len(row), 12) if row[i:i + 12].strip()]

    # SIMIND's one-based settings use 120 views, a 360-degree orbit, and a
    # 2 cm body-contour gap (42 = -2). Each head observes the full source for
    # 10/256 seconds per view; the 90-degree head offset is set at index 41.
    # The normalized source controls Monte Carlo histories, while index 25
    # carries the physical activity and exposure.
    parameters = {
        1: 140.5, 2: 37.638, 3: 31.488, 4: 31.488, 5: 37.638,
        8: 25, 10: 20, 12: 25, 14: -7, 15: -7, 22: 9, 23: 0.38,
        24: 0.891, 25: activity_mbq * 10.0 / PHASES,
        28: 0.35, 29: 120, 30: 360, 31: 0.246, 32: 0,
        33: 1, 34: 306, 38: 0.246, 41: head * 90, 42: -2,
        76: 128, 77: 128, 78: 256, 79: 256, 81: 256, 82: 256, 84: 1,
    }
    for index, value in parameters.items():
        values[index - 1] = value
    lines[start:stop] = ["".join(f"{value:12.5E}" for value in values[row:row + 5])
                         for row in range(0, 120, 5)]

    file_row = next(i for i, line in enumerate(lines) if "# Data files" in line) + 1
    lines[file_row + 4] = "source"
    lines[file_row + 5] = "source"
    flag_row = next(i for i, line in enumerate(lines) if "# Simulation flags" in line) + 1
    flags = list(lines[flag_row])
    flags[7] = "T"  # use the configured random-number generator state
    lines[flag_row] = "".join(flags)
    (run_dir / "cardiac.smc").write_text("\n".join(lines) + "\n")
    (run_dir / "cardiac.win").write_text(
        "126.45,154.55,0\n119.425,126.45,0\n154.55,161.575,0\n")


def make_run(phase: int, head: int, args: argparse.Namespace) -> Path:
    xcat = args.xcat_dir
    activity_path = xcat / f"cardiac256_act_{phase}.bin"
    attenuation_path = xcat / f"cardiac256_atn_{phase}.bin"
    if not activity_path.is_file() or not attenuation_path.is_file():
        raise FileNotFoundError(f"phase {phase}: expected {activity_path.name} and {attenuation_path.name} in {xcat}")

    activity = np.fromfile(activity_path, dtype="<f4")
    attenuation_bytes = attenuation_path.stat().st_size
    expected_bytes = VOXELS * 4
    if activity.size != VOXELS or attenuation_bytes != expected_bytes:
        raise ValueError(f"phase {phase}: native files must contain {VOXELS} float32 voxels")
    total = float(activity.astype(np.float64).sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"phase {phase}: activity must have a positive finite sum")
    activity_mbq = total * SOURCE_SCALE_MBQ

    run_dir = args.output_dir / f"phase_{phase:03d}" / f"head_{head}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "projection_tot_w1.a00").exists():
        raise FileExistsError(f"SIMIND output already exists in {run_dir}")
    scaled = (activity * (args.histories / total)).astype("<f4")
    scaled.tofile(run_dir / "source_act_1.bin")
    target_atn = run_dir / "source_atn_1.bin"
    if target_atn.exists() or target_atn.is_symlink():
        target_atn.unlink()
    target_atn.symlink_to(attenuation_path.resolve())
    configure_smc(args.template, run_dir, head, activity_mbq)

    # Two distinct streams per phase; all 512 phase/head combinations are unique.
    run_id = 2 * (phase - 1) + head
    ij, kl = 1802 + run_id, 9373 + 17 * run_id
    data_dir = run_dir / "smc_data"
    data_dir.mkdir(exist_ok=True)
    for item in args.smc_dir.iterdir():
        target = data_dir / item.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if item.is_file():
            if item.suffix in (".num", ".ini"):
                shutil.copy2(item, target)
            else:
                target.symlink_to(item.resolve())
    state_path = data_dir / "ranmar.num"
    state_path.write_text(" ".join(str(value) for value in ranmar_state(ij, kl)) + "\n")
    initial_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    (data_dir / "ranmar_initial.num").write_bytes(state_path.read_bytes())

    if not args.dry_run:
        command = [str(args.simind), "cardiac", "projection",
                   "/px:0.246/th:0.246/sf:1/df:1/cc:gi-lehr/sc:10"]
        environment = os.environ.copy()
        environment["SMC_DIR"] = str(data_dir.resolve()) + os.sep
        subprocess.run(command, cwd=run_dir, env=environment, check=True)
        for window in (1, 2, 3):
            projection = run_dir / f"projection_tot_w{window}.a00"
            if projection.stat().st_size != 128 * 128 * 120 * 4:
                raise RuntimeError(f"Unexpected SIMIND projection size: {projection}")
        (run_dir / "source_act_1.bin").unlink()
        result = {"phase": phase, "head": head, "histories_per_view": args.histories,
                  "activity_mbq": activity_mbq, "exposure_seconds": 10 / PHASES,
                  "ranmar_ij": ij, "ranmar_kl": kl,
                  "ranmar_initial_sha256": initial_hash, "command": command}
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xcat-dir", type=Path, default=HERE.parent / "xcat" / "output",
                        help="directory with cardiac256_{act,atn}_PHASE.bin files")
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--template", type=Path, default=HERE / "simind_template.smc")
    parser.add_argument("--phase", type=int, action="append",
                        help="one-based XCAT phase 1–256; repeat to select several (default: all)")
    parser.add_argument("--histories", type=int, default=500_000,
                        help="source histories per view (default: 500000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="prepare configurations and source files without launching SIMIND")
    args = parser.parse_args()
    args.xcat_dir = args.xcat_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.template = args.template.expanduser().resolve()
    if not os.environ.get("SIMIND_PATH"):
        parser.error("set SIMIND_PATH to your SIMIND installation directory")
    simind_path = Path(os.environ["SIMIND_PATH"]).expanduser().resolve()
    args.simind = simind_path / "simind"
    args.smc_dir = simind_path / "smc_dir"
    phases = args.phase if args.phase else list(range(1, PHASES + 1))
    if any(phase < 1 or phase > PHASES for phase in phases):
        parser.error("phase numbers must be in the range 1–256")
    if args.histories < 1:
        parser.error("histories must be positive")
    if not args.template.is_file():
        parser.error(f"template not found: {args.template}")
    if not args.dry_run and not args.simind.is_file():
        parser.error(f"SIMIND executable not found: {args.simind}")
    if not args.smc_dir.is_dir():
        parser.error(f"SIMIND data directory not found: {args.smc_dir}")
    args.phases = phases
    return args


def main() -> int:
    args = parse_args()
    for phase in args.phases:
        for head in (0, 1):
            run_dir = make_run(phase, head, args)
            print(f"Prepared phase {phase}, head {head}: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
