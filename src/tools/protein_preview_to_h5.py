from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


def xdmf_hyperslab(h5name: str, dataset: str, ii: int, m0: int, m1: int, m2: int) -> str:
    return (
        f"<DataItem ItemType=\"HyperSlab\" Dimensions=\"{m0} {m1} {m2}\" Type=\"HyperSlab\">\n"
        "  <DataItem Dimensions=\"3 4\" Format=\"XML\">\n"
        f"    {ii} 0 0 0\n"
        "    1 1 1 1\n"
        f"    1 {m0} {m1} {m2}\n"
        "  </DataItem>\n"
        f"  <DataItem Dimensions=\"{{strn}} {m0} {m1} {m2}\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">{h5name}:/{dataset}</DataItem>\n"
        "</DataItem>\n"
    )


def write_proteins_xdmf(path: Path, h5name: str, strn: int, m0: int, m1: int, m2: int,
                        d0: float, d1: float, d2: float) -> None:
    dx = d0 / m0
    dy = d1 / m1
    dz = d2 / m2
    parts = [
        "<?xml version=\"1.0\" ?>\n",
        "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n",
        "<Xdmf Version=\"3.0\">\n",
        " <Domain>\n",
        "  <Grid Name=\"Proteins\" GridType=\"Collection\" CollectionType=\"Temporal\">\n",
    ]
    for ii in range(strn):
        parts.append(
            "   <Grid Name=\"replica_{0}\" GridType=\"Uniform\">\n"
            "    <Time Value=\"{0}\"/>\n"
            "    <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"{1} {2} {3}\"/>\n"
            "    <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n"
            "     <DataItem Dimensions=\"3\" Format=\"XML\">0 0 0</DataItem>\n"
            "     <DataItem Dimensions=\"3\" Format=\"XML\">{4:.10g} {5:.10g} {6:.10g}</DataItem>\n"
            "    </Geometry>\n".format(ii, m0, m1, m2, dx, dy, dz)
        )
        for name in ("prot1", "prot2", "prot3"):
            parts.append(f"    <Attribute Name=\"{name}\" AttributeType=\"Scalar\" Center=\"Node\">\n")
            parts.append(xdmf_hyperslab(h5name, name, ii, m0, m1, m2).replace("{strn}", str(strn)))
            parts.append("    </Attribute>\n")
        parts.append("   </Grid>\n")
    parts.extend(["  </Grid>\n", " </Domain>\n", "</Xdmf>\n"])
    path.write_text("".join(parts), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 11:
        print("usage: protein_preview_to_h5.py raw_path out_h5 out_xdmf m0 m1 m2 strn d0 d1 d2", file=sys.stderr)
        return 2

    raw_path = Path(sys.argv[1])
    out_h5 = Path(sys.argv[2])
    out_xdmf = Path(sys.argv[3])
    m0 = int(sys.argv[4])
    m1 = int(sys.argv[5])
    m2 = int(sys.argv[6])
    strn = int(sys.argv[7])
    d0 = float(sys.argv[8])
    d1 = float(sys.argv[9])
    d2 = float(sys.argv[10])

    total = 3 * strn * m0 * m1 * m2
    data = np.fromfile(raw_path, dtype=np.float32)
    if data.size != total:
        raise ValueError(f"expected {total} float32 values in {raw_path}, found {data.size}")
    data = data.reshape(3, strn, m0, m1, m2)

    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset("prot1", data=data[0], compression="gzip", compression_opts=4, shuffle=True)
        h5.create_dataset("prot2", data=data[1], compression="gzip", compression_opts=4, shuffle=True)
        h5.create_dataset("prot3", data=data[2], compression="gzip", compression_opts=4, shuffle=True)

    write_proteins_xdmf(out_xdmf, out_h5.name, strn, m0, m1, m2, d0, d1, d2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
