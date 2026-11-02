#!/usr/bin/env python3
"""Generate one XCAT activity and attenuation volume for each cardiac phase.

Set XCAT_PATH to the XCAT installation directory.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "xcat_template.par"
ASSETS = (
    "xcat2.cfg",
    "vfemale50.nrb",
    "vfemale50_heart.nrb",
    "atten_table.dat",
    "heart_curve.txt",
    "diaphragm_curve.dat",
    "ap_curve.dat",
    "tumor_curve.dat",
)
PHASES = 256


def set_parameter(text: str, key: str, value: str) -> str:
    """Replace a parameter value while preserving its row and comments."""
    pattern = re.compile(rf"^({re.escape(key)}\s*=\s*)\S+", re.MULTILINE)
    text, count = pattern.subn(rf"\g<1>{value}", text)
    if count != 1:
        raise ValueError(f"Expected exactly one '{key}' row in the parameter template; found {count}.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output", help="Directory for generated .bin files")
    parser.add_argument("--first-phase", type=int, default=1, help="First phase to generate (1-256)")
    parser.add_argument("--phase-count", type=int, default=PHASES, help="Number of consecutive phases")
    args = parser.parse_args()

    if not os.environ.get("XCAT_PATH"):
        parser.error("set XCAT_PATH to your XCAT installation directory")
    assets = Path(os.environ["XCAT_PATH"]).expanduser().resolve()
    executable = assets / "dxcat2_64bit"
    output = args.output.expanduser().resolve()
    if not executable.is_file():
        parser.error(f"XCAT executable not found: {executable}")
    if not TEMPLATE.is_file():
        parser.error(f"Parameter template not found: {TEMPLATE}")
    if args.phase_count < 1 or args.first_phase < 1 or args.first_phase + args.phase_count - 1 > PHASES:
        parser.error("Requested phase range must be within 1..256 and contain at least one phase")
    missing = [name for name in ASSETS if not (assets / name).is_file()]
    if missing:
        parser.error(f"Missing XCAT asset files in {assets}: {', '.join(missing)}")

    output.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text()
    for phase in range(args.first_phase, args.first_phase + args.phase_count):
        destinations = [output / f"cardiac256_{kind}_{phase}.bin" for kind in ("act", "atn")]
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing output: {existing[0]}")

        phase_text = set_parameter(template, "hrt_start_ph_index", f"{(phase - 1) / PHASES:.12f}")
        phase_text = set_parameter(phase_text, "out_period", f"{1 / PHASES:.8f}")
        with tempfile.TemporaryDirectory(prefix=f"xcat_phase_{phase:03d}_", dir=output) as work:
            workdir = Path(work)
            for name in ASSETS:
                (workdir / name).symlink_to(assets / name)
            (workdir / "phase.par").write_text(phase_text)
            subprocess.run([str(executable), "phase.par", "cardiac256"], cwd=workdir, check=True)
            generated = [workdir / f"cardiac256_{kind}_1.bin" for kind in ("act", "atn")]
            absent = [path.name for path in generated if not path.is_file()]
            if absent:
                raise RuntimeError(f"XCAT did not create expected output(s): {', '.join(absent)}")
            for source, destination in zip(generated, destinations):
                shutil.move(str(source), destination)
        print(f"Generated phase {phase}/{PHASES}", flush=True)


if __name__ == "__main__":
    main()
