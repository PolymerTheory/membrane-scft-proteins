"""Check full geometry fields against the small example reference."""
from pathlib import Path
import argparse
import json
import h5py
import numpy as np


def verify(directory, reference):
    expected = json.loads(Path(reference).read_text())
    result = {}
    with h5py.File(Path(directory) / "proteins.h5") as h:
        for key, target in expected.items():
            a = h[key][:]
            if list(a.shape) != target["shape"] or not np.isfinite(a).all():
                raise ValueError(f"Invalid geometry field {key}")
            stats = {"min": float(a.min()), "max": float(a.max()),
                     "sum": float(a.sum(dtype=np.float64))}
            for name, value in stats.items():
                if not np.isclose(value, target[name], rtol=1e-5, atol=1e-6):
                    raise ValueError(f"{key}/{name}: {value} differs from {target[name]}")
            result[key] = {"shape": list(a.shape), **stats}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("reference")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.reference), indent=2))
