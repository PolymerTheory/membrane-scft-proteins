"""Verify execution-demo outputs. This is not a scientific convergence test."""
from pathlib import Path
import argparse
import json
import re
import h5py
import numpy as np


def verify(directory):
    directory = Path(directory)
    result = {}
    log = (directory / "solver.log").read_text()
    if not re.search(r"^Done \(0\)", log, re.MULTILINE):
        raise ValueError("solver.log must record normal Done exit for rank 0")
    groups = {"wins.h5": ("W1", "W2", "Wp"),
              "concentrations.h5": ("rhoA", "rhoB", "rhoH"),
              "proteins.h5": ("prot1", "prot2", "prot3")}
    for name, keys in groups.items():
        with h5py.File(directory / name) as h:
            for key in keys:
                values = h[key][:]
                if values.shape != (4, 32, 32, 32) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {name}/{key}: {values.shape}")
                result[f"{name}/{key}"] = {"shape": list(values.shape),
                    "min": float(values.min()), "max": float(values.max())}
    energy = np.loadtxt(directory / "FEs")
    if energy.ndim != 2 or energy.shape[0] != 4 or not np.isfinite(energy).all():
        raise ValueError("FEs must contain four finite replica rows")
    result["FEs"] = energy.tolist()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
