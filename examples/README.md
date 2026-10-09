# Demonstrations

Run these commands from the repository root. Each demo includes `input.dat`,
`prot_input.dat`, positions (`P0s`) and orientations (`qtns`); no external data
files are needed. Output belongs in separate `build/` directories.

| Folder | Purpose | Hardware | Expected output |
|---|---|---|---|
| `geometry` | Paired arc protein fields, additive reflection | CPU | `proteins.h5`, `proteins.xdmf` |
| `single-protein` | Single arc protein fields, reflection off | CPU | `proteins.h5`, `proteins.xdmf` |
| `gpu-smoke` | Bounded SCFT solver execution | One NVIDIA GPU | Fields, densities, proteins, free energies |

Python dependencies: `python -m pip install -r requirements.txt`.
The GPU solver additionally requires CUDA/CUFFT, MPI and HDF5 development libraries.

## Paired protein geometry

```sh
python -m pip install -r requirements.txt
make -C src protein_preview SCFT_STRN=4
mkdir -p build/geometry
cp examples/geometry/*.dat examples/geometry/P0s examples/geometry/qtns build/geometry/
(cd build/geometry && ../../src/protein_preview input.dat)
```

Expected: `proteins.h5` and `proteins.xdmf`, containing finite protein fields
`prot1`, `prot2`, `prot3` for four 32x32x32 replicas. Additive mode yields
z-reflection symmetry. Use `--vtk` to export twelve VTK files instead.
The clean preview build took 1.48 seconds and this demo took 1.17 seconds
on the tested Mac, with the compiler and Python dependencies already installed.
Dependency installation is additional and was not timed in a fresh environment.
Expected field maxima and sums are in `examples/geometry/expected_output.json`;
small floating-point differences across platforms are possible.
Preview uses unit chi; solver fields include the input interaction strength.

## GPU execution

Adapt MPI compiler and library locations to your system.

```sh
make -C src SCFT_STRN=4 USE_HDF5=1 NVCC='nvcc -ccbin mpicxx'
mkdir -p build/gpu-smoke
cp examples/gpu-smoke/*.dat examples/gpu-smoke/P0s examples/gpu-smoke/qtns build/gpu-smoke/
(cd build/gpu-smoke && mpiexec -n 1 ../../src/scft_gpu input.dat > solver.log 2>&1)
python examples/gpu-smoke/check_output.py build/gpu-smoke
```

Expected: finite `FEs`, restart `wins.h5`, density `concentrations.h5`, protein
`proteins.h5`, corresponding visualization metadata and normal `Done` exit.
One GPU/rank can own the four replicas. The clean public-source GPU smoke
ran in 17 seconds of solver time (20.53 seconds including launch) on one
NVIDIA GH200 GPU on JUPITER; NVIDIA driver 595.71.05. Full output arrays
and four free-energy rows were finite, and the solver exited normally.
Reference output statistics and free-energy rows are in
`examples/gpu-smoke/expected_output.json` for inspection; numerical values
may differ across platforms. This five-iteration analytical-start demo verifies execution. Its unconverged
densities can be negative or exceed physical bounds; use the output to check
installation, not as a physical-density reference solution.


Desktop execution (GTX 1060 6GB) took **110 s solver time / 113.45 s including
launch**, with clean MPI exit and full finite-field/energy checks. The successful
source build took **12.17 s** with dependencies already installed.
Desktop reference statistics are in `gpu-smoke/expected_output_desktop.json`.
CPU geometry on the same desktop took 0.34 s (paired) and 0.28 s (single);
preview compilation took 2.38 s. Dependency installation is additional.

## Single protein geometry

After building `protein_preview` as above:

```sh
mkdir -p build/single-protein
cp examples/single-protein/*.dat examples/single-protein/P0s examples/single-protein/qtns build/single-protein/
(cd build/single-protein && ../../src/protein_preview input.dat)
python examples/check_geometry.py build/single-protein examples/single-protein/expected_output.json
```

This uses the same four orientations with reflection disabled. The preview
writes protein geometry with unit interaction strength; it does not solve SCFT.

## Checking outputs

```sh
python examples/check_geometry.py build/geometry examples/geometry/expected_output.json
python examples/gpu-smoke/check_output.py build/gpu-smoke
```

Geometry checks compare full-array shapes, finiteness and reference extrema/sums.
The GPU checker validates full-array shapes, finiteness, four free-energy rows
and normal solver exit. A successful execution test does not establish physical
convergence. Inspect residuals and energy stability for scientific calculations.

## Linux desktop build example

For a CUDA 12.0 installation with headers in `/usr/include` and a GTX 1060,
use `sm_61`. This example assumes an activated conda-forge environment providing
GCC/G++ 12, Open MPI, HDF5, zlib, Python, NumPy and h5py:

```sh
conda create -n membrane-scft -c conda-forge python=3.12 numpy h5py hdf5 \
  openmpi gcc_linux-64=12 gxx_linux-64=12 make zlib
conda activate membrane-scft
export OMPI_CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
make -C src SCFT_STRN=4 USE_HDF5=1 HDF5_DIR="$CONDA_PREFIX" \
  NVCC='nvcc -ccbin mpicxx -arch=sm_61 -I/usr/include' \
  LDFLAGS="-L$CONDA_PREFIX/lib -L/usr/lib/x86_64-linux-gnu"
```

Use the GPU execution commands above after building. With WSL2, the NVIDIA
Windows driver must expose the GPU to Linux. CUDA headers/toolkit are needed
for building; the driver-reported maximum CUDA version is not the installed
compiler version. Match your compiler and GPU architecture to your CUDA toolkit.

## Tested environments

- CPU: macOS 27.0.1 arm64, Apple clang 21.0.0, Python 3.12.9,
  NumPy 2.5.0, h5py 3.13.0 (HDF5 1.14.6).
- GPU: Linux aarch64, RHEL 9.8, NVIDIA GH200, driver 595.71.05,
  GCC 14.3.0, ParaStationMPI 5.13.0-1, CUDA 13.0, HDF5 1.14.6.

- Desktop GPU and CPU: NVIDIA GeForce GTX 1060 6GB, AMD Ryzen 7 5700G,
  23 GiB RAM available to Linux, Ubuntu 24.04.2 x86_64 under WSL2;
  driver 576.80, CUDA 12.0 (nvcc 12.0.140), GCC 12.4.0, Open MPI 5.0.11,
  HDF5 2.2.0, Python 3.12.15, NumPy 2.5.3, h5py 3.16.0.

Timings above exclude fresh dependency installation. Numerical values can differ
across hardware and compiler versions. Large production grids require more memory
and time than these four-replica 32³ demos.
