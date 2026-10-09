# Examples and installation checks

The examples answer two questions: **are the protein fields being constructed
correctly?** and **does the CUDA/MPI solver run and write usable output files?**
They use a small periodic box of side length 8 (model units), a 32³ grid and
four replicas with protein tilts of 0, 0.2, 0.4 and 0.6 radians about y.
Each folder supplies its own parameters, positions and quaternions; no downloads
or external restart files are needed. Run commands from the repository root.

| Example | System | What it tests | What it does not test |
|---|---|---|---|
| [geometry](geometry/) | Capped semicircular arc plus its additive z-reflection; two patches at ±20° | Protein-field generation for paired scaffolds at four tilts | Membrane response or SCFT convergence |
| [single-protein](single-protein/) | The same arc and patches, with reflection off | Single-scaffold geometry at the same tilts | Membrane response or SCFT convergence |
| [gpu-smoke](gpu-smoke/) | Diblock/solvent fields coupled to the paired protein geometry | GPU field/density calculation, MPI execution and output writing | Equilibrium membrane structure, a fission barrier or string-path convergence |

The GPU example starts from analytical fields (`readwin=0`), with a radial
perturbation in replica 0 and uniform fields in the others. It performs one
outer step with five composition iterations, holds proteins fixed and disables
string updates. It is an installation check, not a converged protein-on-tube
example. Densities at this early stage can be negative or exceed physical bounds.

## Dependencies: solver versus optional tools

- **CUDA solver:** CUDA/CUFFT, MPI, C++ compiler, HDF5 development libraries
  and zlib. It does not call Python.
- **CPU geometry preview:** C++ compiler. Its default HDF5 export calls
  `src/tools/protein_preview_to_h5.py`; NumPy reads the temporary array dump
  and h5py writes HDF5. `--vtk` writes VTK directly in C++, without Python.
- **Optional checkers and tests:** Python, NumPy and h5py. NumPy checks array
  shapes, finite values and statistics; h5py reads the HDF5 datasets.

Install the optional Python tools with:

```sh
python -m pip install -r requirements.txt
```

## 1. Protein geometry on a CPU

Build the preview once, then run both examples:

```sh
make -C src protein_preview SCFT_STRN=4
mkdir -p build/geometry build/single-protein
cp examples/geometry/*.dat examples/geometry/P0s examples/geometry/qtns build/geometry/
cp examples/single-protein/*.dat examples/single-protein/P0s examples/single-protein/qtns build/single-protein/
(cd build/geometry && ../../src/protein_preview input.dat)
(cd build/single-protein && ../../src/protein_preview input.dat)
python examples/check_geometry.py build/geometry examples/geometry/expected_output.json
python examples/check_geometry.py build/single-protein examples/single-protein/expected_output.json
```

Both write `proteins.h5` and `proteins.xdmf`: three fields (`prot1`, `prot2`,
`prot3`), each of shape `(4, 32, 32, 32)`. Open the XDMF file in a compatible
viewer to inspect the geometry. The paired example adds reflected potentials,
including their overlap; the single example does not reflect them. Both use
aligned end caps and midpoint alignment during tilting.

The checker reads every array value and compares shape, finiteness, extrema
and sums with the bundled references. These are aggregate numerical checks;
they do not establish point-by-point equality or physical convergence.
The preview uses unit `chi`; solver protein potentials are scaled by `chi[0]`.

For VTK output without Python, append `--vtk` to a preview invocation. This
writes twelve files (three fields × four replicas); the Python checkers above
expect HDF5, so skip them for VTK-only output.

## 2. Short CUDA/MPI solver run

Build for your GPU and library installation:

```sh
make -C src SCFT_STRN=4 USE_HDF5=1 NVCC='nvcc -ccbin mpicxx'
mkdir -p build/gpu-smoke
cp examples/gpu-smoke/*.dat examples/gpu-smoke/P0s examples/gpu-smoke/qtns build/gpu-smoke/
(cd build/gpu-smoke && mpiexec -n 1 ../../src/scft_gpu input.dat > solver.log 2>&1)
python examples/gpu-smoke/check_output.py build/gpu-smoke
```

Use one MPI rank/GPU for the four replicas. Set `HDF5_DIR` if your HDF5
installation is not found automatically. The checker is optional: the solver
run itself needs no Python. The MPI launcher must return exit status 0.

Expected files and checks:

- `solver.log`: the normal rank-0 `Done` message.
- `wins.h5`: three restart fields (`W1`, `W2`, `Wp`).
- `concentrations.h5`: three density fields (`rhoA`, `rhoB`, `rhoH`).
- `proteins.h5`: three protein potentials (`prot1`, `prot2`, `prot3`).
- `FEs`: four finite free-energy rows; XDMF files describe the grid for viewing.

The checker reads all nine arrays, verifies shape `(4, 32, 32, 32)` and
finiteness, and checks the free-energy rows and `Done` message. It does not
relaunch from a checkpoint, measure scientific convergence or compare energies
to a fixed tolerance. The two hardware reference JSON files below record observed
values for inspection; cross-platform numerical values can differ.

## Measured example runs

These are two distinct GPU environments running the same small example.
Times exclude fresh dependency installation and do not predict production runtime.

### Desktop: NVIDIA GeForce GTX 1060, 6 GB

- Ubuntu 24.04.2 x86_64 under WSL2; AMD Ryzen 7 5700G; 23 GiB RAM available to Linux.
- Driver 576.80; CUDA 12.0 (`nvcc` 12.0.140); GCC 12.4.0; Open MPI 5.0.11;
  HDF5 2.2.0.
- Optional tools: Python 3.12.15, NumPy 2.5.3, h5py 3.16.0.
- GPU build: **12.17 s**. Solver: **110 s**; MPI launch through exit: **113.45 s**.
- CPU preview build: **2.38 s**; paired preview: **0.34 s**; single preview: **0.28 s**.
- Result: clean MPI exit; all output arrays and free energies finite; both CPU
  previews matched their reference statistics.
- GPU reference: [expected_output_desktop.json](gpu-smoke/expected_output_desktop.json).

The following build recipe was tested with CUDA headers in `/usr/include` and
CUDA libraries in `/usr/lib/x86_64-linux-gnu`. Adjust paths for other installations.
CUDA itself must already be installed; conda supplies the remaining tools below.
Python/NumPy/h5py are included for exporting and checking the examples.

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

Then use the run commands in section 2. Under WSL2, the Windows NVIDIA driver
must expose the GPU to Linux. The driver-reported maximum CUDA version is not
the installed toolkit version; match the toolkit, compiler and GPU architecture.

### JUPITER: NVIDIA GH200

- Linux aarch64, RHEL 9.8; driver 595.71.05; CUDA 13.0; GCC 14.3.0;
  ParaStationMPI 5.13.0-1; HDF5 1.14.6.
- Solver: **17 s**; launch through exit: **20.53 s** on one GH200 GPU.
- Result: normal solver exit; all output arrays and free energies finite.
- GPU reference: [expected_output.json](gpu-smoke/expected_output.json).

This cluster run predates the addition of normal-exit `MPI_Finalize`; the
GTX 1060 run also verified that change with a clean MPI launcher exit. These
measurements are separate installation checks, not a controlled hardware benchmark.

### Additional CPU check: macOS arm64

macOS 27.0.1, Apple clang 21.0.0; Python 3.12.9, NumPy 2.5.0,
h5py 3.13.0 (HDF5 1.14.6). Paired preview build: **1.48 s**; run: **1.17 s**.

## Optional automated tests

```sh
python -m unittest discover -s tests
```

The tests exercise geometry switches (including additive reflection and caps)
and check that output-validation helpers reject malformed or nonfinite results.
They complement the executable examples; they do not validate a physical pathway.
